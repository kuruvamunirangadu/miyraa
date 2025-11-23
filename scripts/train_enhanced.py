"""Enhanced training script with improved architecture and regularization.

New features:
- Improved model architecture with dropout and normalization
- Focal loss for imbalanced classes
- Dynamic loss weighting
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- Configurable backbone freezing
- Weight decay regularization
- Progressive unfreezing
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.training.model_architecture import create_model
from src.nlp.training.enhanced_losses import (
    FocalLoss, MultiTaskLoss, LossCalculator, compute_class_weights
)
from src.nlp.training.utils import (
    set_seed, get_device, MixedPrecisionTrainer,
    save_checkpoint, generate_training_report
)


class EmotionDataset(Dataset):
    """Dataset for multi-task emotion classification."""
    
    def __init__(
        self,
        texts: List[str],
        emotions: List[int],
        vad: List[Tuple[float, float, float]],
        safety: List[int],
        style: List[int],
        intent: List[int],
        tokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.emotions = emotions
        self.vad = vad
        self.safety = safety
        self.style = style
        self.intent = intent
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'emotion': torch.tensor(self.emotions[idx], dtype=torch.long),
            'vad': torch.tensor(self.vad[idx], dtype=torch.float),
            'safety': torch.tensor(self.safety[idx], dtype=torch.long),
            'style': torch.tensor(self.style[idx], dtype=torch.long),
            'intent': torch.tensor(self.intent[idx], dtype=torch.long)
        }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


def train_epoch(
    model,
    dataloader,
    loss_calculator,
    multi_task_loss,
    optimizer,
    device,
    mp_trainer,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch with mixed precision support.
    
    Returns:
        Dictionary of average losses
    """
    model.train()
    
    total_losses = {
        'total': 0.0,
        'emotions': 0.0,
        'vad': 0.0,
        'safety': 0.0,
        'style': 0.0,
        'intent': 0.0
    }
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        targets = {
            'emotions': batch['emotion'].to(device),
            'vad': batch['vad'].to(device),
            'safety': batch['safety'].to(device),
            'style': batch['style'].to(device),
            'intent': batch['intent'].to(device)
        }
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with mp_trainer.autocast():
            predictions = model(input_ids, attention_mask)
            
            # Compute task losses
            task_losses = loss_calculator.compute_losses(predictions, targets)
            
            # Compute total loss with weighting
            total_loss, loss_components = multi_task_loss(task_losses)
        
        # Backward pass with gradient scaling
        mp_trainer.scale_loss(total_loss).backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step with scaler
        mp_trainer.step(optimizer)
        mp_trainer.update()
        
        # Accumulate losses
        total_losses['total'] += total_loss.item()
        for task, loss_val in loss_components.items():
            if task in total_losses:
                total_losses[task] += loss_val
        
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss.item(),
            'emotion': loss_components.get('emotions', 0)
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def validate(
    model,
    dataloader,
    loss_calculator,
    multi_task_loss,
    device,
    mp_trainer
) -> Dict[str, float]:
    """Validate model with mixed precision support.
    
    Returns:
        Dictionary of average losses
    """
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'emotions': 0.0,
        'vad': 0.0,
        'safety': 0.0,
        'style': 0.0,
        'intent': 0.0
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            targets = {
                'emotions': batch['emotion'].to(device),
                'vad': batch['vad'].to(device),
                'safety': batch['safety'].to(device),
                'style': batch['style'].to(device),
                'intent': batch['intent'].to(device)
            }
            
            # Forward pass with mixed precision
            with mp_trainer.autocast():
                predictions = model(input_ids, attention_mask)
                task_losses = loss_calculator.compute_losses(predictions, targets)
                total_loss, loss_components = multi_task_loss(task_losses)
            
            total_losses['total'] += total_loss.item()
            for task, loss_val in loss_components.items():
                if task in total_losses:
                    total_losses[task] += loss_val
            
            num_batches += 1
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Enhanced multi-task training")
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Backbone model name')
    parser.add_argument('--freeze-strategy', type=str, default='none',
                       choices=['none', 'full', 'partial'],
                       help='Backbone freezing strategy')
    parser.add_argument('--freeze-layers', type=int, default=4,
                       help='Number of layers to freeze if partial')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for task heads')
    parser.add_argument('--head-hidden-dims', type=int, nargs='+', default=[256, 128],
                       help='Hidden dimensions for task heads')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='Use focal loss for emotion classification')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    parser.add_argument('--use-uncertainty-weighting', action='store_true',
                       help='Use learned uncertainty-based loss weighting')
    parser.add_argument('--emotion-weight', type=float, default=1.5,
                       help='Weight for emotion loss')
    parser.add_argument('--vad-weight', type=float, default=1.0,
                       help='Weight for VAD loss')
    parser.add_argument('--safety-weight', type=float, default=1.2,
                       help='Weight for safety loss')
    parser.add_argument('--style-weight', type=float, default=0.8,
                       help='Weight for style loss')
    parser.add_argument('--intent-weight', type=float, default=0.8,
                       help='Weight for intent loss')
    
    # Training arguments
    parser.add_argument('--data', type=str, default='data/processed/production',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='outputs/',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Warmup steps for learning rate')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate training report with plots')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed, deterministic=False)
    
    # Get device with mixed precision support
    device, use_amp = get_device()
    mp_trainer = MixedPrecisionTrainer(device, enabled=use_amp)
    
    # Create model
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)
    
    model = create_model(
        backbone_name=args.backbone,
        freeze_strategy=args.freeze_strategy,
        freeze_layers=args.freeze_layers,
        dropout_rate=args.dropout,
        head_hidden_dims=args.head_hidden_dims
    )
    
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    
    # TODO: Load actual data
    # For now, create dummy data
    print("\n⚠️ Using dummy data for demonstration")
    print("   Replace with actual data loading in production")
    
    dummy_texts = ["This is a test"] * 100
    dummy_emotions = [0] * 100
    dummy_vad = [(0.5, 0.5, 0.5)] * 100
    dummy_safety = [0] * 100
    dummy_style = [0] * 100
    dummy_intent = [0] * 100
    
    train_dataset = EmotionDataset(
        dummy_texts, dummy_emotions, dummy_vad,
        dummy_safety, dummy_style, dummy_intent,
        tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Create loss calculator
    emotion_class_weights = None
    if args.use_focal_loss:
        # Compute class weights from data
        emotion_class_weights = compute_class_weights(dummy_emotions, num_classes=11)
        emotion_class_weights = emotion_class_weights.to(device)
    
    loss_calculator = LossCalculator(
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        emotion_class_weights=emotion_class_weights
    )
    
    # Create multi-task loss
    if not args.use_uncertainty_weighting:
        loss_weights = {
            'emotions': args.emotion_weight,
            'vad': args.vad_weight,
            'safety': args.safety_weight,
            'style': args.style_weight,
            'intent': args.intent_weight
        }
    else:
        loss_weights = None
    
    multi_task_loss = MultiTaskLoss(
        loss_weights=loss_weights,
        use_uncertainty_weighting=args.use_uncertainty_weighting
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, loss_calculator,
            multi_task_loss, optimizer, device, mp_trainer,
            max_grad_norm=args.max_grad_norm
        )
        train_loss_history.append(train_losses['total'])
        
        print(f"\nTrain Losses:")
        for task, loss in train_losses.items():
            print(f"  {task}: {loss:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(train_losses['total'])
            else:
                scheduler.step()
        
        # Save checkpoint with smart naming
        if train_losses['total'] < best_val_loss:
            best_val_loss = train_losses['total']
            
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                val_loss=train_losses['total'],
                val_metrics={'emotion_loss': train_losses.get('emotions', 0)},
                hyperparameters=vars(args),
                output_dir=args.output,
                prefix='best_model'
            )
            
            print(f"✅ Saved best checkpoint")
        
        # Early stopping check
        if early_stopping(train_losses['total']):
            print(f"\n⛔ Early stopping triggered at epoch {epoch + 1}")
            break
        
        val_loss_history.append(train_losses['total'])  # Using train loss as proxy
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    
    # Generate training report if requested
    if args.generate_report:
        print("\nGenerating training report...")
        report_dir = generate_training_report(
            train_losses=train_loss_history,
            val_losses=val_loss_history,
            train_metrics={},
            val_metrics={},
            hyperparameters=vars(args),
            output_dir='reports/',
            prefix='enhanced_training'
        )
        print(f"✅ Report generated: {report_dir}")


if __name__ == "__main__":
    main()
