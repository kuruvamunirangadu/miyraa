"""
Evaluate Model Calibration and Generate Thresholds

This script:
1. Loads a trained model checkpoint
2. Evaluates calibration on validation set
3. Generates reliability diagrams
4. Optimizes per-class thresholds
5. Creates confusion matrices and classification reports

Usage:
    python scripts/evaluate_calibration.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap
    
    # With custom output directory
    python scripts/evaluate_calibration.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --output reports/calibration
    
    # Optimize for precision instead of F1
    python scripts/evaluate_calibration.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --metric precision

Author: Miyraa Team
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.training.evaluation import (
    CalibrationAnalyzer,
    ThresholdOptimizer,
    ConfusionMatrixAnalyzer
)
from src.nlp.models.multi_task_model import MultiTaskModel


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """
    Load model checkpoint and tokenizer.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model
        tokenizer: Tokenizer
        metadata: Checkpoint metadata
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    metadata = checkpoint.get('metadata', {})
    backbone_name = metadata.get('backbone_name', 'xlm-roberta-base')
    
    print(f"  Backbone: {backbone_name}")
    print(f"  Training seed: {metadata.get('seed', 'N/A')}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    # Create model
    model = MultiTaskModel(
        backbone_name=backbone_name,
        num_emotion_classes=5,
        num_sentiment_classes=3,
        num_intent_classes=10,
        num_topic_classes=8,
        num_safety_classes=2
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    
    return model, tokenizer, metadata


def prepare_dataloader(data_dir: str, tokenizer, batch_size: int = 32):
    """
    Prepare validation dataloader.
    
    Args:
        data_dir: Directory containing processed data
        tokenizer: Tokenizer
        batch_size: Batch size
    
    Returns:
        dataloader: PyTorch DataLoader
        dataset: Raw dataset for label access
    """
    from torch.utils.data import DataLoader
    
    print(f"Loading validation data from: {data_dir}")
    
    # Load dataset
    dataset_dict = load_from_disk(data_dir)
    val_dataset = dataset_dict['validation'] if 'validation' in dataset_dict else dataset_dict['train']
    
    print(f"  Samples: {len(val_dataset)}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    tokenized = val_dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'emotion'])
    
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=False)
    
    print("✓ Dataloader prepared\n")
    
    return dataloader, val_dataset


def collect_predictions(model, dataloader, device):
    """
    Collect all predictions on validation set.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device
    
    Returns:
        y_true: True labels
        y_probs: Predicted probabilities
        y_pred: Predicted labels
    """
    print("Collecting predictions...")
    
    y_true_list = []
    y_probs_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}", end='\r')
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['emotion']
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emotion_logits = outputs['emotion']
            
            # Convert to probabilities
            probs = torch.softmax(emotion_logits, dim=-1).cpu().numpy()
            
            y_true_list.append(labels.numpy())
            y_probs_list.append(probs)
    
    y_true = np.concatenate(y_true_list)
    y_probs = np.concatenate(y_probs_list)
    y_pred = np.argmax(y_probs, axis=1)
    
    print(f"\n✓ Collected {len(y_true)} predictions\n")
    
    return y_true, y_probs, y_pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate model calibration')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='reports/calibration',
                        help='Output directory for reports')
    parser.add_argument('--metric', type=str, default='f1',
                        choices=['f1', 'precision', 'recall'],
                        help='Metric to optimize thresholds for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MODEL CALIBRATION EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Threshold metric: {args.metric}")
    print("="*60 + "\n")
    
    # Load model
    model, tokenizer, metadata = load_model_and_tokenizer(args.checkpoint, device)
    
    # Prepare data
    dataloader, dataset = prepare_dataloader(args.data, tokenizer, args.batch_size)
    
    # Collect predictions
    y_true, y_probs, y_pred = collect_predictions(model, dataloader, device)
    
    # Define class names
    class_names = ['joy', 'sadness', 'anger', 'fear', 'neutral']
    
    # 1. Calibration Analysis
    print("="*60)
    print("1. CALIBRATION ANALYSIS")
    print("="*60)
    
    calibration_analyzer = CalibrationAnalyzer(n_bins=10, class_names=class_names)
    calibration_analyzer.add_predictions(y_true, y_probs)
    
    ece = calibration_analyzer.compute_ece()
    mce = calibration_analyzer.compute_mce()
    
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Maximum Calibration Error (MCE): {mce:.4f}")
    
    # Plot reliability diagram
    calibration_analyzer.plot_reliability_diagram(
        str(output_dir / 'reliability_diagram.png'),
        title='Model Calibration - Reliability Diagram'
    )
    
    # Plot per-class calibration
    calibration_analyzer.plot_per_class_calibration(
        str(output_dir / 'per_class_calibration.png'),
        num_classes=5
    )
    
    print()
    
    # 2. Threshold Optimization
    print("="*60)
    print("2. THRESHOLD OPTIMIZATION")
    print("="*60)
    
    threshold_optimizer = ThresholdOptimizer(class_names=class_names)
    thresholds = threshold_optimizer.optimize_thresholds(
        y_true, y_probs, metric=args.metric, num_classes=5
    )
    
    print(f"Optimal thresholds (optimized for {args.metric}):")
    for class_name, threshold in thresholds.items():
        print(f"  {class_name:10s}: {threshold:.3f}")
    
    # Save thresholds
    threshold_metadata = {
        'metric': args.metric,
        'n_samples': len(y_true),
        'ece': float(ece),
        'mce': float(mce),
        'checkpoint': args.checkpoint,
        'timestamp': timestamp
    }
    
    threshold_optimizer.save_thresholds(
        str(output_dir / 'thresholds.yaml'),
        thresholds,
        metadata=threshold_metadata
    )
    
    # Also save to main outputs directory
    threshold_optimizer.save_thresholds(
        'outputs/thresholds.yaml',
        thresholds,
        metadata=threshold_metadata
    )
    
    print()
    
    # 3. Confusion Matrix Analysis
    print("="*60)
    print("3. CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    cm_analyzer = ConfusionMatrixAnalyzer(class_names=class_names)
    
    # Plot confusion matrix
    cm_analyzer.plot_confusion_matrix(
        y_true, y_pred,
        str(output_dir / 'confusion_matrix.png'),
        normalize=True
    )
    
    # Find most confused pairs
    confused_pairs = cm_analyzer.find_most_confused_pairs(y_true, y_pred, top_k=5)
    print("Most confused class pairs:")
    for i, (true_class, pred_class, count) in enumerate(confused_pairs, 1):
        print(f"  {i}. {true_class:10s} → {pred_class:10s}: {count:4d} times")
    
    # Generate classification report
    report = cm_analyzer.generate_classification_report(
        y_true, y_pred,
        output_path=str(output_dir / 'classification_report.txt')
    )
    
    print("\nClassification Report:")
    print(report)
    
    # 4. Save summary
    print("="*60)
    print("4. SAVING SUMMARY")
    print("="*60)
    
    summary = {
        'checkpoint': args.checkpoint,
        'data': args.data,
        'timestamp': timestamp,
        'n_samples': len(y_true),
        'calibration': {
            'ece': float(ece),
            'mce': float(mce)
        },
        'thresholds': thresholds,
        'threshold_metric': args.metric,
        'confused_pairs': [
            {
                'true_class': true_class,
                'pred_class': pred_class,
                'count': count
            }
            for true_class, pred_class, count in confused_pairs
        ],
        'model_metadata': metadata
    }
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - reliability_diagram.png")
    print("  - per_class_calibration.png")
    print("  - thresholds.yaml")
    print("  - confusion_matrix.png")
    print("  - classification_report.txt")
    print("  - evaluation_summary.json")
    print("\n✓ Thresholds also copied to: outputs/thresholds.yaml")


if __name__ == "__main__":
    main()
