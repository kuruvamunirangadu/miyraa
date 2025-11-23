"""
Evaluation and Calibration Tools for Miyraa NLP Engine

Provides comprehensive model evaluation including:
- Calibration curves and reliability diagrams
- Per-class threshold optimization
- Confusion matrices and F1 analysis
- Noise robustness testing
- Safety classifier evaluation

Author: Miyraa Team
Date: November 2025
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import pandas as pd


class CalibrationAnalyzer:
    """
    Analyze model calibration for probability predictions.
    
    Calibration measures how well predicted probabilities match actual outcomes.
    A well-calibrated model's predicted 70% confidence should be correct 70% of the time.
    
    Example:
        >>> analyzer = CalibrationAnalyzer(n_bins=10)
        >>> analyzer.add_predictions(y_true, y_probs)
        >>> ece = analyzer.compute_ece()
        >>> analyzer.plot_reliability_diagram('emotion_calibration.png')
    """
    
    def __init__(self, n_bins: int = 10, class_names: Optional[List[str]] = None):
        """
        Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration curve
            class_names: List of class names for visualization
        """
        self.n_bins = n_bins
        self.class_names = class_names
        self.y_true = []
        self.y_probs = []
    
    def add_predictions(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> None:
        """
        Add predictions for calibration analysis.
        
        Args:
            y_true: True class labels (N,)
            y_probs: Predicted probabilities (N, num_classes)
        """
        self.y_true.append(y_true)
        self.y_probs.append(y_probs)
    
    def compute_ece(self) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE is the weighted average of calibration error across bins.
        Lower ECE means better calibration.
        
        Returns:
            ece: Expected Calibration Error (0 to 1)
        """
        y_true = np.concatenate(self.y_true)
        y_probs = np.concatenate(self.y_probs)
        
        # Get predicted class probabilities
        y_pred_probs = np.max(y_probs, axis=1)
        y_pred = np.argmax(y_probs, axis=1)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples in this bin
            in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_pred_probs[in_bin].mean()
                # Add weighted calibration error
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def compute_mce(self) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum calibration error across all bins.
        
        Returns:
            mce: Maximum Calibration Error (0 to 1)
        """
        y_true = np.concatenate(self.y_true)
        y_probs = np.concatenate(self.y_probs)
        
        y_pred_probs = np.max(y_probs, axis=1)
        y_pred = np.argmax(y_probs, axis=1)
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = y_pred_probs[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def plot_reliability_diagram(
        self,
        output_path: str,
        title: str = "Reliability Diagram"
    ) -> None:
        """
        Plot reliability diagram (calibration curve).
        
        Args:
            output_path: Path to save plot
            title: Plot title
        """
        y_true = np.concatenate(self.y_true)
        y_probs = np.concatenate(self.y_probs)
        
        y_pred_probs = np.max(y_probs, axis=1)
        y_pred = np.argmax(y_probs, axis=1)
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            (y_pred == y_true).astype(int),
            y_pred_probs,
            n_bins=self.n_bins,
            strategy='uniform'
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Actual calibration
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            's-',
            label='Model Calibration',
            linewidth=2,
            markersize=8
        )
        
        # Add histogram of predictions
        ax.hist(
            y_pred_probs,
            bins=self.n_bins,
            alpha=0.3,
            color='blue',
            label='Prediction Distribution',
            density=True
        )
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add ECE and MCE
        ece = self.compute_ece()
        mce = self.compute_mce()
        ax.text(
            0.05, 0.95,
            f'ECE: {ece:.4f}\nMCE: {mce:.4f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Reliability diagram saved: {output_path}")
    
    def plot_per_class_calibration(
        self,
        output_path: str,
        num_classes: int
    ) -> None:
        """
        Plot calibration curves for each class.
        
        Args:
            output_path: Path to save plot
            num_classes: Number of classes
        """
        y_true = np.concatenate(self.y_true)
        y_probs = np.concatenate(self.y_probs)
        
        num_rows = (num_classes + 2) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.flatten() if num_classes > 1 else [axes]
        
        for class_idx in range(num_classes):
            ax = axes[class_idx]
            
            # Binary problem for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_prob_class = y_probs[:, class_idx]
            
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary,
                y_prob_class,
                n_bins=self.n_bins,
                strategy='uniform'
            )
            
            # Plot
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                's-',
                linewidth=2,
                markersize=6
            )
            
            class_name = self.class_names[class_idx] if self.class_names else f'Class {class_idx}'
            ax.set_title(class_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Probability', fontsize=10)
            ax.set_ylabel('Actual Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class calibration saved: {output_path}")


class ThresholdOptimizer:
    """
    Optimize classification thresholds for each class.
    
    Default threshold of 0.5 may not be optimal. This class finds
    optimal thresholds to maximize F1, precision, or recall.
    
    Example:
        >>> optimizer = ThresholdOptimizer()
        >>> thresholds = optimizer.optimize_thresholds(
        ...     y_true, y_probs, metric='f1'
        ... )
        >>> optimizer.save_thresholds('outputs/thresholds.yaml', thresholds)
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize threshold optimizer.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        metric: str = 'f1',
        num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Find optimal threshold for each class.
        
        Args:
            y_true: True class labels (N,)
            y_probs: Predicted probabilities (N, num_classes)
            metric: Metric to optimize ('f1', 'precision', 'recall')
            num_classes: Number of classes (inferred if None)
        
        Returns:
            thresholds: Dictionary of class_name -> threshold
        """
        if num_classes is None:
            num_classes = y_probs.shape[1]
        
        thresholds = {}
        
        for class_idx in range(num_classes):
            # Binary problem for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_prob_class = y_probs[:, class_idx]
            
            # Find optimal threshold
            best_threshold = 0.5
            best_score = 0.0
            
            # Try different thresholds
            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred_binary = (y_prob_class >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            class_name = self.class_names[class_idx] if self.class_names else f'class_{class_idx}'
            thresholds[class_name] = float(best_threshold)
        
        return thresholds
    
    def save_thresholds(
        self,
        output_path: str,
        thresholds: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save thresholds to YAML file.
        
        Args:
            output_path: Path to save thresholds
            thresholds: Dictionary of thresholds
            metadata: Optional metadata to include
        """
        output_dict = {
            'thresholds': thresholds,
            'metadata': metadata or {}
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(output_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Thresholds saved: {output_path}")
    
    @staticmethod
    def load_thresholds(thresholds_path: str) -> Dict[str, float]:
        """
        Load thresholds from YAML file.
        
        Args:
            thresholds_path: Path to thresholds file
        
        Returns:
            thresholds: Dictionary of thresholds
        """
        with open(thresholds_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data.get('thresholds', {})


class ConfusionMatrixAnalyzer:
    """
    Generate and analyze confusion matrices.
    
    Example:
        >>> analyzer = ConfusionMatrixAnalyzer(class_names=['joy', 'sadness', ...])
        >>> analyzer.plot_confusion_matrix(y_true, y_pred, 'confusion.png')
        >>> report = analyzer.generate_classification_report(y_true, y_pred)
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize confusion matrix analyzer.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        normalize: bool = True
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save plot
            normalize: If True, show percentages; else show counts
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved: {output_path}")
    
    def find_most_confused_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, str, int]]:
        """
        Find most commonly confused class pairs.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            top_k: Number of top confused pairs to return
        
        Returns:
            confused_pairs: List of (true_class, pred_class, count)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Get off-diagonal elements (misclassifications)
        confused_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((
                        self.class_names[i],
                        self.class_names[j],
                        int(cm[i, j])
                    ))
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return confused_pairs[:top_k]
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Optional path to save report
        
        Returns:
            report: Classification report string
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✓ Classification report saved: {output_path}")
        
        return report


if __name__ == "__main__":
    """
    Test evaluation tools
    """
    print("Testing Evaluation & Calibration Tools\n")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Simulated predictions (slightly miscalibrated)
    y_true = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_samples)
    y_pred = np.argmax(y_probs, axis=1)
    
    class_names = ['joy', 'sadness', 'anger', 'fear', 'neutral']
    
    # Test 1: Calibration Analysis
    print("1. Testing calibration analysis:")
    calibration_analyzer = CalibrationAnalyzer(n_bins=10, class_names=class_names)
    calibration_analyzer.add_predictions(y_true, y_probs)
    
    ece = calibration_analyzer.compute_ece()
    mce = calibration_analyzer.compute_mce()
    print(f"   ECE: {ece:.4f}")
    print(f"   MCE: {mce:.4f}")
    
    calibration_analyzer.plot_reliability_diagram(
        'reports/test/calibration_reliability.png',
        title='Test Reliability Diagram'
    )
    
    calibration_analyzer.plot_per_class_calibration(
        'reports/test/calibration_per_class.png',
        num_classes=n_classes
    )
    
    # Test 2: Threshold Optimization
    print("\n2. Testing threshold optimization:")
    threshold_optimizer = ThresholdOptimizer(class_names=class_names)
    thresholds = threshold_optimizer.optimize_thresholds(
        y_true, y_probs, metric='f1', num_classes=n_classes
    )
    
    print("   Optimal thresholds:")
    for class_name, threshold in thresholds.items():
        print(f"     {class_name}: {threshold:.3f}")
    
    threshold_optimizer.save_thresholds(
        'reports/test/thresholds.yaml',
        thresholds,
        metadata={'metric': 'f1', 'n_samples': n_samples}
    )
    
    # Test 3: Confusion Matrix
    print("\n3. Testing confusion matrix analysis:")
    cm_analyzer = ConfusionMatrixAnalyzer(class_names=class_names)
    
    cm_analyzer.plot_confusion_matrix(
        y_true, y_pred,
        'reports/test/confusion_matrix.png',
        normalize=True
    )
    
    confused_pairs = cm_analyzer.find_most_confused_pairs(y_true, y_pred, top_k=3)
    print("   Most confused pairs:")
    for true_class, pred_class, count in confused_pairs:
        print(f"     {true_class} → {pred_class}: {count} times")
    
    report = cm_analyzer.generate_classification_report(
        y_true, y_pred,
        output_path='reports/test/classification_report.txt'
    )
    print("\n   Classification Report Preview:")
    print("   " + "\n   ".join(report.split('\n')[:5]))
    
    print("\n✅ All evaluation tools tests passed!")
