"""
Training Module for Miyraa NLP Engine

Provides training-specific functionality:
- Enhanced loss functions (focal loss, multi-task loss)
- Training utilities (mixed precision, checkpointing, reporting)
- Model architecture (moved to src.nlp.models for sharing)

Separated from inference code for clean organization.

Author: Miyraa Team
Date: November 2025
"""

from .enhanced_losses import (
    FocalLoss,
    MultiTaskLoss,
    LossCalculator,
    compute_class_weights,
    find_optimal_loss_weights,
)

from .utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    generate_training_report,
    MixedPrecisionTrainer,
    print_reproducibility_info,
)

# Import model architecture from shared models module
# This maintains backward compatibility while using the shared location
try:
    from ..models import ImprovedTaskHead, MultiTaskModel, create_model
except ImportError:
    # Fallback to local if models package not available
    from .model_architecture import ImprovedTaskHead, MultiTaskModel, create_model

__all__ = [
    # Loss functions
    'FocalLoss',
    'MultiTaskLoss',
    'LossCalculator',
    'compute_class_weights',
    'find_optimal_loss_weights',
    
    # Training utilities
    'set_seed',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    'generate_training_report',
    'MixedPrecisionTrainer',
    'print_reproducibility_info',
    
    # Model architecture (from shared models)
    'ImprovedTaskHead',
    'MultiTaskModel',
    'create_model',
]
