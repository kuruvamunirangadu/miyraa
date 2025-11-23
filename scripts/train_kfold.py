"""
K-Fold Cross-Validation Training Script for Miyraa NLP Engine

Performs k-fold cross-validation with stratified splits for classification tasks.
Computes per-fold metrics and aggregates results with mean and standard deviation.

Usage:
    python scripts/train_kfold.py --data data/processed/production --k 5 --epochs 10

Author: Miyraa Team
Date: November 2025
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Dict, List, Tuple, Any
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.training.model_architecture import create_model, MultiTaskModel
from src.nlp.training.enhanced_losses import LossCalculator, MultiTaskLoss, compute_class_weights
from src.nlp.training.utils import (
    set_seed, get_device, save_checkpoint, generate_training_report,
    MixedPrecisionTrainer
)
from transformers import AutoTokenizer


class EmotionDataset(Dataset):
    """Multi-task emotion dataset for k-fold cross-validation"""
    
    def __init__(self, texts: List[str], labels: Dict[str, List], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Add task-specific labels
        for task_name, task_labels in self.labels.items():
            if task_name == 'vad':
                # Regression task (3 values)
                item[task_name] = torch.tensor(task_labels[idx], dtype=torch.float32)
            else:
                # Classification task
                item[task_name] = torch.tensor(task_labels[idx], dtype=torch.long)
        
        return item


def load_data(data_path: str) -> Tuple[List[str], Dict[str, List]]:
    """
    Load training data from JSONL file.
    
    Args:
        data_path: Path to data directory or JSONL file
    
    Returns:
        texts: List of text samples
        labels: Dictionary of task_name -> list of labels
    """
    # Find JSONL file
    if os.path.isdir(data_path):
        jsonl_files = list(Path(data_path).glob('*.jsonl'))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {data_path}")
        data_file = jsonl_files[0]
    else:
        data_file = data_path
    
    print(f"Loading data from: {data_file}")
    
    texts = []
    emotions = []
    vad_values = []
    styles = []
    intents = []
    safety = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            texts.append(sample['text'])
            emotions.append(sample['emotion_id'])
            vad_values.append(sample['vad'])
            styles.append(sample.get('style_id', 0))
            intents.append(sample.get('intent_id', 0))
            safety.append(sample.get('safety_id', 0))
    
    labels = {
        'emotions': emotions,
        'vad': vad_values,
        'style': styles,
        'intent': intents,
        'safety': safety,
    }
    
    print(f"✓ Loaded {len(texts)} samples")
    return texts, labels


def train_epoch(
    model: MultiTaskModel,
    dataloader: DataLoader,
    loss_calculator: LossCalculator,
    multi_task_loss: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mp_trainer: MixedPrecisionTrainer,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        metrics: Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch in dataloader:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = {k: v.to(device) for k, v in batch.items() 
                  if k not in ['input_ids', 'attention_mask']}
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with mp_trainer.autocast():
            outputs = model(input_ids, attention_mask)
            task_losses = loss_calculator.compute_losses(outputs, targets)
            loss, _ = multi_task_loss(task_losses)
        
        # Backward pass with gradient scaling
        mp_trainer.scale_loss(loss).backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        mp_trainer.step(optimizer)
        mp_trainer.update()
        
        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    return {'loss': avg_loss}


def validate(
    model: MultiTaskModel,
    dataloader: DataLoader,
    loss_calculator: LossCalculator,
    multi_task_loss: MultiTaskLoss,
    device: torch.device,
    mp_trainer: MixedPrecisionTrainer
) -> Dict[str, float]:
    """
    Validate model.
    
    Returns:
        metrics: Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() 
                      if k not in ['input_ids', 'attention_mask']}
            
            # Forward pass with mixed precision
            with mp_trainer.autocast():
                outputs = model(input_ids, attention_mask)
                task_losses = loss_calculator.compute_losses(outputs, targets)
                loss, _ = multi_task_loss(task_losses)
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    return {'loss': avg_loss}


def train_fold(
    fold_idx: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    mp_trainer: MixedPrecisionTrainer
) -> Dict[str, Any]:
    """
    Train a single fold.
    
    Returns:
        results: Dictionary containing fold results
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # For reproducibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = create_model(
        backbone_name=args.backbone,
        freeze_strategy=args.freeze_strategy,
        freeze_layers=args.freeze_layers,
        dropout_rate=args.dropout
    ).to(device)
    
    # Compute class weights for focal loss
    emotion_labels = [train_dataset[i]['emotions'].item() for i in range(len(train_dataset))]
    class_weights = compute_class_weights(emotion_labels, num_classes=11) if args.use_focal_loss else None
    
    # Create loss calculator
    loss_calculator = LossCalculator(
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        emotion_class_weights=class_weights.to(device) if class_weights is not None else None
    )
    
    # Create multi-task loss
    multi_task_loss = MultiTaskLoss(
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        num_tasks=5 if args.use_uncertainty_weighting else None
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5
        )
    else:
        scheduler = None
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_calculator, multi_task_loss,
            optimizer, device, mp_trainer, args.max_grad_norm
        )
        train_losses.append(train_metrics['loss'])
        
        # Validate
        val_metrics = validate(
            model, val_loader, loss_calculator, multi_task_loss,
            device, mp_trainer
        )
        val_losses.append(val_metrics['loss'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        
        # Update best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            print(f"✓ New best validation loss: {best_val_loss:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
    
    return {
        'fold': fold_idx + 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory or JSONL file')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of folds (default: 5)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified k-fold (based on emotion labels)')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Transformer backbone model')
    parser.add_argument('--freeze-strategy', type=str, default='partial',
                       choices=['full', 'partial', 'none'],
                       help='Backbone freezing strategy')
    parser.add_argument('--freeze-layers', type=int, default=4,
                       help='Number of layers to freeze (for partial strategy)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate in task heads')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='Use focal loss for emotion classification')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--use-uncertainty-weighting', action='store_true',
                       help='Use uncertainty-based multi-task loss weighting')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per fold')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='reports/kfold',
                       help='Output directory for results')
    parser.add_argument('--max-length', type=int, default=128,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed, deterministic=True)
    
    # Get device
    device, use_amp = get_device()
    mp_trainer = MixedPrecisionTrainer(device, enabled=use_amp)
    
    # Load data
    texts, labels = load_data(args.data)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    
    # Create full dataset
    full_dataset = EmotionDataset(texts, labels, tokenizer, args.max_length)
    
    # Create k-fold splitter
    if args.stratified:
        kfold = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        split_labels = labels['emotions']  # Use emotion labels for stratification
        print(f"\n✓ Using Stratified K-Fold Cross-Validation (k={args.k})")
    else:
        kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        split_labels = list(range(len(texts)))  # Dummy labels for non-stratified
        print(f"\n✓ Using K-Fold Cross-Validation (k={args.k})")
    
    # Store results for all folds
    all_fold_results = []
    
    # Train each fold
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(texts, split_labels)):
        # Create fold datasets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # Train fold
        fold_results = train_fold(
            fold_idx, train_dataset, val_dataset,
            args, device, mp_trainer
        )
        
        all_fold_results.append(fold_results)
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    
    best_val_losses = [r['best_val_loss'] for r in all_fold_results]
    final_val_losses = [r['final_val_loss'] for r in all_fold_results]
    
    print(f"\nBest Validation Loss per Fold:")
    for i, loss in enumerate(best_val_losses, 1):
        print(f"  Fold {i}: {loss:.4f}")
    
    print(f"\nAggregated Results:")
    print(f"  Mean Best Val Loss:   {np.mean(best_val_losses):.4f} ± {np.std(best_val_losses):.4f}")
    print(f"  Mean Final Val Loss:  {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
    print(f"  Min Best Val Loss:    {np.min(best_val_losses):.4f}")
    print(f"  Max Best Val Loss:    {np.max(best_val_losses):.4f}")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join(args.output, f"kfold_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'hyperparameters': vars(args),
            'fold_results': all_fold_results,
            'aggregated': {
                'mean_best_val_loss': float(np.mean(best_val_losses)),
                'std_best_val_loss': float(np.std(best_val_losses)),
                'mean_final_val_loss': float(np.mean(final_val_losses)),
                'std_final_val_loss': float(np.std(final_val_losses)),
                'min_best_val_loss': float(np.min(best_val_losses)),
                'max_best_val_loss': float(np.max(best_val_losses)),
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"{'='*80}\n")
    
    print("✅ K-Fold Cross-Validation Complete!")


if __name__ == "__main__":
    main()
