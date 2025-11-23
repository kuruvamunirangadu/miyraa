"""
Safety Classifier Evaluation for Miyraa NLP Engine

Evaluates safety classifier performance with focus on:
- False Positives: Safe content incorrectly flagged as unsafe
- False Negatives: Unsafe content that passes through
- Cost-weighted metrics (FN more expensive than FP)
- Threshold recommendations

Usage:
    python scripts/evaluate_safety.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap
    
    # With custom FN cost multiplier
    python scripts/evaluate_safety.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --fn-cost 5.0
    
    # Test threshold sweep
    python scripts/evaluate_safety.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --threshold-sweep

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
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.models.multi_task_model import MultiTaskModel


class SafetyEvaluator:
    """
    Evaluate safety classifier with emphasis on FP/FN analysis.
    
    Example:
        >>> evaluator = SafetyEvaluator(fn_cost=5.0)
        >>> metrics = evaluator.evaluate(y_true, y_probs)
        >>> evaluator.plot_threshold_analysis(y_true, y_probs, 'threshold_analysis.png')
    """
    
    def __init__(self, fn_cost: float = 3.0):
        """
        Initialize safety evaluator.
        
        Args:
            fn_cost: Cost multiplier for false negatives vs false positives
                     (e.g., 3.0 means FN is 3x worse than FP)
        """
        self.fn_cost = fn_cost
    
    def compute_cost_weighted_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute cost-weighted score considering FN cost.
        
        Args:
            y_true: True binary labels (0=safe, 1=unsafe)
            y_pred: Predicted binary labels
        
        Returns:
            cost_score: Weighted cost (lower is better)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # cm[1,0] = False Negatives (unsafe predicted as safe) - HIGH COST
        # cm[0,1] = False Positives (safe predicted as unsafe) - LOW COST
        
        fp = cm[0, 1] if cm.shape[0] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 else 0
        
        total_cost = fp + (fn * self.fn_cost)
        
        return total_cost
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        metric: str = 'cost'
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold for safety classification.
        
        Args:
            y_true: True binary labels (0=safe, 1=unsafe)
            y_probs: Predicted probabilities for unsafe class
            metric: Metric to optimize ('cost', 'f1', 'precision', 'recall')
        
        Returns:
            best_threshold: Optimal threshold
            metrics: Dictionary of metrics at optimal threshold
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_score = float('inf') if metric == 'cost' else 0.0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            
            if metric == 'cost':
                score = self.compute_cost_weighted_score(y_true, y_pred)
                is_better = score < best_score
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
                is_better = score > best_score
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
                is_better = score > best_score
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
                is_better = score > best_score
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if is_better:
                best_score = score
                best_threshold = threshold
                
                # Compute all metrics at this threshold
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cost = self.compute_cost_weighted_score(y_true, y_pred)
                
                cm = confusion_matrix(y_true, y_pred)
                fp = int(cm[0, 1]) if cm.shape[0] > 1 else 0
                fn = int(cm[1, 0]) if cm.shape[0] > 1 else 0
                
                best_metrics = {
                    'threshold': float(threshold),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'cost': float(cost),
                    'fp_count': fp,
                    'fn_count': fn
                }
        
        return best_threshold, best_metrics
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        output_path: str
    ) -> None:
        """
        Plot threshold sweep analysis.
        
        Args:
            y_true: True binary labels
            y_probs: Predicted probabilities
            output_path: Path to save plot
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        
        precisions = []
        recalls = []
        f1_scores = []
        costs = []
        fp_counts = []
        fn_counts = []
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cost = self.compute_cost_weighted_score(y_true, y_pred)
            
            cm = confusion_matrix(y_true, y_pred)
            fp = int(cm[0, 1]) if cm.shape[0] > 1 else 0
            fn = int(cm[1, 0]) if cm.shape[0] > 1 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            costs.append(cost)
            fp_counts.append(fp)
            fn_counts.append(fn)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Precision, Recall, F1
        ax1 = axes[0, 0]
        ax1.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2, markersize=6)
        ax1.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2, markersize=6)
        ax1.plot(thresholds, f1_scores, 'g-^', label='F1', linewidth=2, markersize=6)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cost
        ax2 = axes[0, 1]
        ax2.plot(thresholds, costs, 'purple', linewidth=2.5, marker='o', markersize=6)
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel(f'Weighted Cost (FN cost = {self.fn_cost}x)', fontsize=12)
        ax2.set_title('Cost-Weighted Score vs Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal point
        min_cost_idx = np.argmin(costs)
        ax2.plot(thresholds[min_cost_idx], costs[min_cost_idx], 'r*', markersize=20,
                label=f'Optimal (t={thresholds[min_cost_idx]:.2f})')
        ax2.legend(fontsize=11)
        
        # Plot 3: FP vs FN counts
        ax3 = axes[1, 0]
        ax3.plot(thresholds, fp_counts, 'orange', label='False Positives', linewidth=2, marker='o')
        ax3.plot(thresholds, fn_counts, 'red', label='False Negatives', linewidth=2, marker='s')
        ax3.set_xlabel('Threshold', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('FP vs FN Count', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Precision-Recall Curve
        ax4 = axes[1, 1]
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        auc_pr = auc(recall_curve, precision_curve)
        ax4.plot(recall_curve, precision_curve, 'b-', linewidth=2.5)
        ax4.set_xlabel('Recall', fontsize=12)
        ax4.set_ylabel('Precision', fontsize=12)
        ax4.set_title(f'Precision-Recall Curve (AUC={auc_pr:.3f})', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Threshold analysis saved: {output_path}")
    
    def analyze_edge_cases(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        num_examples: int = 10
    ) -> Dict:
        """
        Find and analyze edge cases (borderline predictions).
        
        Args:
            texts: Input texts
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities
            num_examples: Number of examples to return per category
        
        Returns:
            edge_cases: Dictionary of edge case examples
        """
        # Find false positives
        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_indices = np.where(fp_mask)[0]
        fp_probs = y_probs[fp_indices]
        
        # Sort by confidence (high confidence FPs are most problematic)
        fp_sorted = fp_indices[np.argsort(-fp_probs)][:num_examples]
        
        # Find false negatives
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_indices = np.where(fn_mask)[0]
        fn_probs = y_probs[fn_indices]
        
        # Sort by confidence (high confidence FNs are most problematic)
        fn_sorted = fn_indices[np.argsort(fn_probs)][:num_examples]
        
        # Find borderline cases (predictions near 0.5)
        borderline_mask = np.abs(y_probs - 0.5) < 0.1
        borderline_indices = np.where(borderline_mask)[0][:num_examples]
        
        edge_cases = {
            'false_positives': [
                {
                    'text': texts[idx],
                    'true_label': 'safe',
                    'predicted_label': 'unsafe',
                    'confidence': float(y_probs[idx])
                }
                for idx in fp_sorted
            ],
            'false_negatives': [
                {
                    'text': texts[idx],
                    'true_label': 'unsafe',
                    'predicted_label': 'safe',
                    'confidence': float(1 - y_probs[idx])
                }
                for idx in fn_sorted
            ],
            'borderline_cases': [
                {
                    'text': texts[idx],
                    'true_label': 'safe' if y_true[idx] == 0 else 'unsafe',
                    'predicted_label': 'safe' if y_pred[idx] == 0 else 'unsafe',
                    'confidence': float(max(y_probs[idx], 1 - y_probs[idx]))
                }
                for idx in borderline_indices
            ]
        }
        
        return edge_cases


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load model checkpoint and tokenizer."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    metadata = checkpoint.get('metadata', {})
    backbone_name = metadata.get('backbone_name', 'xlm-roberta-base')
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    model = MultiTaskModel(
        backbone_name=backbone_name,
        num_emotion_classes=5,
        num_sentiment_classes=3,
        num_intent_classes=10,
        num_topic_classes=8,
        num_safety_classes=2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    
    return model, tokenizer


def collect_predictions(model, dataloader, device):
    """Collect all safety predictions."""
    print("Collecting safety predictions...")
    
    y_true_list = []
    y_probs_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}", end='\r')
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['safety']
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            safety_logits = outputs['safety']
            
            # Convert to probabilities (probability of unsafe class)
            probs = torch.softmax(safety_logits, dim=-1)[:, 1].cpu().numpy()
            
            y_true_list.append(labels.numpy())
            y_probs_list.append(probs)
    
    y_true = np.concatenate(y_true_list)
    y_probs = np.concatenate(y_probs_list)
    
    print(f"\n✓ Collected {len(y_true)} predictions\n")
    
    return y_true, y_probs


def main():
    parser = argparse.ArgumentParser(description='Evaluate safety classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='reports/safety',
                        help='Output directory for reports')
    parser.add_argument('--fn-cost', type=float, default=3.0,
                        help='Cost multiplier for false negatives (default: 3.0)')
    parser.add_argument('--threshold-sweep', action='store_true',
                        help='Perform threshold sweep analysis')
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
    print("SAFETY CLASSIFIER EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"FN Cost Multiplier: {args.fn_cost}x")
    print("="*60 + "\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    
    # Prepare data
    from torch.utils.data import DataLoader
    
    print(f"Loading validation data...")
    dataset_dict = load_from_disk(args.data)
    val_dataset = dataset_dict['validation'] if 'validation' in dataset_dict else dataset_dict['train']
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    tokenized = val_dataset.map(tokenize_function, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'safety'])
    
    dataloader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ Loaded {len(val_dataset)} samples\n")
    
    # Collect predictions
    y_true, y_probs = collect_predictions(model, dataloader, device)
    
    # Initialize evaluator
    evaluator = SafetyEvaluator(fn_cost=args.fn_cost)
    
    # 1. Find optimal threshold
    print("="*60)
    print("1. OPTIMAL THRESHOLD ANALYSIS")
    print("="*60)
    
    optimal_threshold, optimal_metrics = evaluator.find_optimal_threshold(
        y_true, y_probs, metric='cost'
    )
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall: {optimal_metrics['recall']:.4f}")
    print(f"  F1: {optimal_metrics['f1']:.4f}")
    print(f"  Cost: {optimal_metrics['cost']:.2f}")
    print(f"  False Positives: {optimal_metrics['fp_count']}")
    print(f"  False Negatives: {optimal_metrics['fn_count']}")
    
    # 2. Threshold sweep analysis
    if args.threshold_sweep:
        print("\n" + "="*60)
        print("2. THRESHOLD SWEEP ANALYSIS")
        print("="*60)
        
        evaluator.plot_threshold_analysis(
            y_true, y_probs,
            str(output_dir / 'threshold_analysis.png')
        )
    
    # 3. Edge cases analysis
    print("\n" + "="*60)
    print("3. EDGE CASES ANALYSIS")
    print("="*60)
    
    texts = val_dataset['text']
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    edge_cases = evaluator.analyze_edge_cases(texts, y_true, y_pred, y_probs)
    
    print(f"False Positives: {len(edge_cases['false_positives'])}")
    print(f"False Negatives: {len(edge_cases['false_negatives'])}")
    print(f"Borderline Cases: {len(edge_cases['borderline_cases'])}")
    
    # Save edge cases
    edge_cases_path = output_dir / 'edge_cases.json'
    with open(edge_cases_path, 'w') as f:
        json.dump(edge_cases, f, indent=2)
    print(f"\n✓ Edge cases saved: {edge_cases_path}")
    
    # 4. Save summary
    print("\n" + "="*60)
    print("4. SAVING SUMMARY")
    print("="*60)
    
    summary = {
        'checkpoint': args.checkpoint,
        'timestamp': timestamp,
        'n_samples': len(y_true),
        'fn_cost': args.fn_cost,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'edge_cases_counts': {
            'false_positives': len(edge_cases['false_positives']),
            'false_negatives': len(edge_cases['false_negatives']),
            'borderline_cases': len(edge_cases['borderline_cases'])
        }
    }
    
    summary_path = output_dir / 'safety_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved: {summary_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print("\nRecommendation:")
    if optimal_metrics['fn_count'] > optimal_metrics['fp_count'] * 2:
        print("  ⚠ High false negative rate. Consider lowering threshold.")
    elif optimal_metrics['fp_count'] > optimal_metrics['fn_count'] * 5:
        print("  ⚠ High false positive rate. Consider raising threshold.")
    else:
        print("  ✓ Threshold appears well-balanced.")


if __name__ == "__main__":
    main()
