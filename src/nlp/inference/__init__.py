"""
Inference Module for Miyraa NLP Engine

Provides production-ready inference capabilities:
- EmotionInferenceEngine: Main inference class
- ONNX and PyTorch model support
- Batch inference
- Result formatting

Separated from training code for clean deployment.

Author: Miyraa Team
Date: November 2025
"""

from .engine import EmotionInferenceEngine, load_engine
from .dummy_engine import get_engine

__all__ = [
    'EmotionInferenceEngine',
    'load_engine',
    'get_engine',  # Legacy support
]
