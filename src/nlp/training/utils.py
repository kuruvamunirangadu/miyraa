"""
Training Utilities for Miyraa NLP Engine

Provides essential training utilities including:
- Mixed precision training support (fp16)
- Reproducibility (seed setting)
- Checkpoint management with metadata
- Training report generation with visualizations

Author: Miyraa Team
Date: November 2025
"""

import os
import json
import torch
import numpy as np
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and python.random.
    
    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    
    Note:
        Even with seeds set, some operations may not be fully deterministic:
        - CUDA operations (especially on different GPU architectures)
        - DataLoader with num_workers > 0
        - Some PyTorch operations (e.g., torch.nn.functional.interpolate)
    
    Example:
        >>> set_seed(42, deterministic=True)
        >>> # All random operations now reproducible
    """
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            # Make CUDA operations deterministic (can be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Enable cudnn autotuner for better performance
            torch.backends.cudnn.benchmark = True
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed}")
    if deterministic:
        print("✓ Deterministic mode enabled (CUDA operations are reproducible but slower)")


def get_device(prefer_gpu: bool = True) -> Tuple[torch.device, bool]:
    """
    Get the best available device for training.
    
    Args:
        prefer_gpu: If True, use GPU if available; otherwise use CPU
    
    Returns:
        device: torch.device object
        use_amp: Whether to use automatic mixed precision (True for GPU, False for CPU)
    
    Example:
        >>> device, use_amp = get_device()
        >>> model.to(device)
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        use_amp = True
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✓ Mixed Precision (fp16): Enabled")
    else:
        device = torch.device('cpu')
        use_amp = False
        if prefer_gpu and not torch.cuda.is_available():
            print("⚠ GPU requested but not available, using CPU")
        else:
            print("✓ Using CPU (fp32)")
        print(f"✓ Mixed Precision (fp16): Disabled (CPU only supports fp32)")
    
    return device, use_amp


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    val_loss: float,
    val_metrics: Dict[str, float],
    hyperparameters: Dict[str, Any],
    output_dir: str,
    prefix: str = "checkpoint"
) -> str:
    """
    Save model checkpoint with timestamp, metrics, and metadata.
    
    Checkpoint filename format: {prefix}_{timestamp}_{epoch}_{val_loss:.4f}.pt
    
    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        val_loss: Validation loss for this checkpoint
        val_metrics: Dictionary of validation metrics
        hyperparameters: Dictionary of hyperparameters used for training
        output_dir: Directory to save checkpoint
        prefix: Checkpoint filename prefix (default: "checkpoint")
    
    Returns:
        checkpoint_path: Full path to saved checkpoint
    
    Example:
        >>> path = save_checkpoint(
        ...     model, optimizer, scheduler, epoch=10,
        ...     val_loss=0.3456, val_metrics={'accuracy': 0.85},
        ...     hyperparameters={'lr': 2e-5, 'batch_size': 16},
        ...     output_dir='outputs/'
        ... )
        >>> print(f"Saved: {path}")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format filename with timestamp, epoch, and validation loss
    filename = f"{prefix}_{timestamp}_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
    checkpoint_path = os.path.join(output_dir, filename)
    
    # Get git commit hash if available (for reproducibility)
    git_hash = None
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except:
        pass
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'hyperparameters': hyperparameters,
        'timestamp': timestamp,
        'git_hash': git_hash,
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✓ Checkpoint saved: {filename}")
    print(f"  Epoch: {epoch}, Val Loss: {val_loss:.4f}")
    if val_metrics:
        print(f"  Metrics: {', '.join(f'{k}={v:.4f}' for k, v in val_metrics.items())}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        device: Device to load checkpoint to (optional)
    
    Returns:
        metadata: Dictionary containing epoch, metrics, hyperparameters, etc.
    
    Example:
        >>> metadata = load_checkpoint(
        ...     'outputs/checkpoint_20251123_120000_epoch010_loss0.3456.pt',
        ...     model, optimizer, scheduler, device
        ... )
        >>> print(f"Resuming from epoch {metadata['epoch']}")
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', float('inf')),
        'val_metrics': checkpoint.get('val_metrics', {}),
        'hyperparameters': checkpoint.get('hyperparameters', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'git_hash': checkpoint.get('git_hash', None),
    }
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Epoch: {metadata['epoch']}, Val Loss: {metadata['val_loss']:.4f}")
    
    return metadata


def generate_training_report(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    hyperparameters: Dict[str, Any],
    output_dir: str,
    prefix: str = "training_report"
) -> str:
    """
    Generate comprehensive training report with loss curves and metrics summary.
    
    Creates:
    - Loss curves plot (train vs validation)
    - Metrics plots for each tracked metric
    - JSON report with all data
    - Hyperparameters summary
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dictionary of metric_name -> list of values per epoch (training)
        val_metrics: Dictionary of metric_name -> list of values per epoch (validation)
        hyperparameters: Dictionary of hyperparameters used
        output_dir: Directory to save report files
        prefix: Filename prefix for report files
    
    Returns:
        report_dir: Path to directory containing all report files
    
    Example:
        >>> report_dir = generate_training_report(
        ...     train_losses=[2.5, 2.1, 1.8, 1.6],
        ...     val_losses=[2.6, 2.2, 1.9, 1.7],
        ...     train_metrics={'accuracy': [0.6, 0.7, 0.75, 0.8]},
        ...     val_metrics={'accuracy': [0.58, 0.68, 0.73, 0.78]},
        ...     hyperparameters={'lr': 2e-5, 'batch_size': 16},
        ...     output_dir='reports/'
        ... )
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_subdir = os.path.join(output_dir, f"{prefix}_{timestamp}")
    os.makedirs(report_subdir, exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # 1. Loss curves plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot_path = os.path.join(report_subdir, 'loss_curves.png')
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    
    # 2. Metrics plots
    all_metrics = set(train_metrics.keys()) | set(val_metrics.keys())
    num_metrics = len(all_metrics)
    
    if num_metrics > 0:
        nrows = (num_metrics + 1) // 2
        ncols = 2 if num_metrics > 1 else 1
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(14, 5 * nrows)
        )
        
        # Ensure axes is always a 1D array
        if num_metrics == 1:
            axes = np.array([axes])
        elif nrows == 1:
            axes = np.array(axes)
        else:
            axes = axes.flatten()
        
        for idx, metric_name in enumerate(sorted(all_metrics)):
            ax = axes[idx]
            
            if metric_name in train_metrics:
                ax.plot(epochs, train_metrics[metric_name], 'b-o',
                       label=f'Train {metric_name}', linewidth=2, markersize=6)
            
            if metric_name in val_metrics:
                ax.plot(epochs, val_metrics[metric_name], 'r-s',
                       label=f'Val {metric_name}', linewidth=2, markersize=6)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        metrics_plot_path = os.path.join(report_subdir, 'metrics_curves.png')
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
    
    # 3. JSON report with all data
    report_data = {
        'timestamp': timestamp,
        'hyperparameters': hyperparameters,
        'training': {
            'epochs': len(train_losses),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'best_epoch': val_losses.index(min(val_losses)) + 1 if val_losses else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
    }
    
    json_path = os.path.join(report_subdir, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # 4. Human-readable summary
    summary_path = os.path.join(report_subdir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING REPORT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 80 + "\n")
        for key, value in sorted(hyperparameters.items()):
            f.write(f"  {key:30s}: {value}\n")
        f.write("\n")
        
        f.write("TRAINING SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Epochs:               {len(train_losses)}\n")
        if train_losses:
            f.write(f"  Final Training Loss:        {train_losses[-1]:.6f}\n")
        if val_losses:
            f.write(f"  Final Validation Loss:      {val_losses[-1]:.6f}\n")
            f.write(f"  Best Validation Loss:       {min(val_losses):.6f}\n")
            f.write(f"  Best Epoch:                 {val_losses.index(min(val_losses)) + 1}\n")
        f.write("\n")
        
        if val_metrics:
            f.write("FINAL VALIDATION METRICS:\n")
            f.write("-" * 80 + "\n")
            for metric_name in sorted(val_metrics.keys()):
                if val_metrics[metric_name]:
                    final_value = val_metrics[metric_name][-1]
                    f.write(f"  {metric_name:30s}: {final_value:.6f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Training report generated: {report_subdir}")
    print(f"  - Loss curves: loss_curves.png")
    if num_metrics > 0:
        print(f"  - Metrics plots: metrics_curves.png")
    print(f"  - JSON data: report.json")
    print(f"  - Summary: summary.txt")
    
    return report_subdir


class MixedPrecisionTrainer:
    """
    Helper class for mixed precision training with automatic fp16/fp32 handling.
    
    Automatically uses torch.cuda.amp on GPU, falls back to fp32 on CPU.
    
    Example:
        >>> trainer = MixedPrecisionTrainer(device)
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     
        ...     # Forward pass with automatic mixed precision
        ...     with trainer.autocast():
        ...         outputs = model(batch)
        ...         loss = criterion(outputs, targets)
        ...     
        ...     # Backward pass with gradient scaling
        ...     trainer.scale_loss(loss).backward()
        ...     trainer.step(optimizer)
        ...     trainer.update()
    """
    
    def __init__(self, device: torch.device, enabled: bool = True):
        """
        Initialize mixed precision trainer.
        
        Args:
            device: torch.device (cuda or cpu)
            enabled: Whether to enable mixed precision (ignored on CPU)
        """
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.enabled = enabled and self.is_cuda
        
        if self.enabled:
            # Create gradient scaler for fp16 training
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed Precision Trainer initialized (fp16 enabled)")
        else:
            self.scaler = None
            if self.is_cuda:
                print("✓ Mixed Precision Trainer initialized (fp16 disabled)")
            else:
                print("✓ Mixed Precision Trainer initialized (CPU mode, fp32 only)")
    
    def autocast(self):
        """
        Context manager for automatic mixed precision forward pass.
        
        Returns:
            Context manager that handles autocast
        """
        if self.enabled:
            return torch.cuda.amp.autocast()
        else:
            # Dummy context manager (no-op)
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient scaling (only on GPU with fp16).
        
        Args:
            loss: Unscaled loss tensor
        
        Returns:
            Scaled loss (or original loss if not using fp16)
        """
        if self.enabled:
            return self.scaler.scale(loss)
        else:
            return loss
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with gradient unscaling.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self) -> None:
        """
        Update gradient scaler (only needed for fp16).
        """
        if self.enabled:
            self.scaler.update()


def print_reproducibility_info():
    """
    Print information about reproducibility and known non-deterministic operations.
    """
    print("\n" + "=" * 80)
    print("REPRODUCIBILITY NOTES")
    print("=" * 80)
    print("✓ Random seeds have been set for torch, numpy, and python.random")
    print("\nKnown sources of non-determinism:")
    print("  1. CUDA operations (especially on different GPU architectures)")
    print("  2. DataLoader with num_workers > 0 (set to 0 for full reproducibility)")
    print("  3. Some PyTorch operations (e.g., scatter_add_, index_add_)")
    print("  4. Multi-threaded CPU operations")
    print("\nFor maximum reproducibility:")
    print("  - Use deterministic=True in set_seed()")
    print("  - Set num_workers=0 in DataLoader")
    print("  - Run on same hardware and CUDA version")
    print("  - Use the same PyTorch version")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """
    Test training utilities
    """
    print("Testing Training Utilities\n")
    
    # Test 1: Seed setting
    print("1. Testing seed setting:")
    set_seed(42, deterministic=False)
    rand1 = torch.rand(3)
    set_seed(42, deterministic=False)
    rand2 = torch.rand(3)
    assert torch.allclose(rand1, rand2), "Seeds not working correctly!"
    print(f"✓ Reproducibility verified: {rand1.tolist()}\n")
    
    # Test 2: Device detection
    print("2. Testing device detection:")
    device, use_amp = get_device()
    print(f"Device: {device}, Use AMP: {use_amp}\n")
    
    # Test 3: Checkpoint saving/loading
    print("3. Testing checkpoint save/load:")
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint_path = save_checkpoint(
        model, optimizer, None,
        epoch=5, val_loss=0.1234,
        val_metrics={'accuracy': 0.85, 'f1': 0.82},
        hyperparameters={'lr': 0.001, 'batch_size': 32},
        output_dir='outputs/test',
        prefix='test_checkpoint'
    )
    
    # Load checkpoint
    model2 = torch.nn.Linear(10, 5)
    optimizer2 = torch.optim.Adam(model2.parameters())
    metadata = load_checkpoint(checkpoint_path, model2, optimizer2, device=device)
    print(f"Loaded metadata: epoch={metadata['epoch']}, val_loss={metadata['val_loss']}\n")
    
    # Test 4: Training report
    print("4. Testing training report generation:")
    report_dir = generate_training_report(
        train_losses=[2.5, 2.1, 1.8, 1.6, 1.5],
        val_losses=[2.6, 2.2, 1.9, 1.7, 1.6],
        train_metrics={'accuracy': [0.6, 0.7, 0.75, 0.8, 0.82]},
        val_metrics={'accuracy': [0.58, 0.68, 0.73, 0.78, 0.80]},
        hyperparameters={'lr': 2e-5, 'batch_size': 16, 'epochs': 5},
        output_dir='reports/test',
        prefix='test_report'
    )
    
    # Test 5: Mixed precision trainer
    print("\n5. Testing mixed precision trainer:")
    trainer = MixedPrecisionTrainer(device, enabled=True)
    print(f"AMP enabled: {trainer.enabled}")
    
    # Print reproducibility info
    print_reproducibility_info()
    
    print("✅ All training utilities tests passed!")
