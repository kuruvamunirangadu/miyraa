"""
Shared Model Architectures for Miyraa NLP Engine

This package contains model architectures used by both training and inference.
Keeps model definitions separate from training-specific and inference-specific code.

Author: Miyraa Team
Date: November 2025
"""

from .multi_task_model import (
    ImprovedTaskHead,
    MultiTaskModel,
    create_model,
)

__all__ = [
    'ImprovedTaskHead',
    'MultiTaskModel',
    'create_model',
]
