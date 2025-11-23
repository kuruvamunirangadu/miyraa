# Code Organization Guide

**Date**: November 2025  
**Purpose**: Clean separation between training, inference, and shared code

---

## Directory Structure

```
src/nlp/
├── models/              # SHARED: Model architectures used by both training & inference
│   ├── __init__.py
│   └── multi_task_model.py
│
├── training/            # TRAINING ONLY: Training-specific code
│   ├── __init__.py
│   ├── enhanced_losses.py    # Focal loss, multi-task loss, loss calculator
│   ├── losses.py             # Legacy loss functions
│   ├── model_architecture.py # (Legacy, use models/ instead)
│   └── utils.py              # Mixed precision, checkpointing, reporting
│
├── inference/           # INFERENCE ONLY: Inference-specific code  
│   ├── __init__.py
│   ├── engine.py             # Production inference engine
│   └── dummy_engine.py       # Legacy dummy engine
│
├── preprocessing/       # SHARED: Text preprocessing utilities
│   ├── __init__.py
│   ├── normalize.py          # Text normalization
│   └── label_mapping.py      # Label mappings
│
└── safety/              # SHARED: Safety and privacy tools
    └── pii_scrub.py          # PII detection and scrubbing
```

---

## Module Responsibilities

### 1. `src/nlp/models/` - Shared Model Architectures

**Purpose**: Model definitions used by both training and inference

**Contents**:
- `multi_task_model.py`: Multi-task model architecture
  - `ImprovedTaskHead`: Enhanced task heads with dropout, layer norm, residual
  - `MultiTaskModel`: Main multi-task model with flexible freezing
  - `create_model()`: Factory function for model creation

**Usage**:
```python
# Training
from src.nlp.models import create_model
model = create_model(freeze_strategy='partial')

# Inference
from src.nlp.models import MultiTaskModel
model = MultiTaskModel(...)
```

**Why Shared?**
- Both training and inference need the same model architecture
- Avoids code duplication
- Single source of truth for model definitions
- Easy to update model across entire codebase

---

### 2. `src/nlp/training/` - Training-Only Code

**Purpose**: Everything needed for model training

**Contents**:

#### `enhanced_losses.py`
- `FocalLoss`: Handles class imbalance
- `MultiTaskLoss`: Fixed and uncertainty-based weighting
- `LossCalculator`: Computes all task losses
- `compute_class_weights()`: Automatic class weight computation
- `find_optimal_loss_weights()`: Dynamic weight tuning

#### `utils.py`
- `set_seed()`: Reproducibility with seed setting
- `get_device()`: Device detection (GPU/CPU)
- `save_checkpoint()`: Smart checkpoint saving with metadata
- `load_checkpoint()`: Checkpoint loading with state restoration
- `generate_training_report()`: Training report with plots
- `MixedPrecisionTrainer`: Mixed precision training helper (fp16)
- `print_reproducibility_info()`: Reproducibility notes

**Usage**:
```python
# Training script
from src.nlp.training import (
    set_seed, get_device, MixedPrecisionTrainer,
    FocalLoss, MultiTaskLoss, save_checkpoint
)

set_seed(42)
device, use_amp = get_device()
mp_trainer = MixedPrecisionTrainer(device)
focal_loss = FocalLoss(gamma=2.0)
```

**Why Training-Only?**
- These utilities are never needed during inference
- Reduces inference dependencies (no matplotlib, no sklearn)
- Keeps inference lightweight and fast
- Clear separation of concerns

---

### 3. `src/nlp/inference/` - Inference-Only Code

**Purpose**: Production inference with ONNX and PyTorch support

**Contents**:

#### `engine.py`
- `EmotionInferenceEngine`: Main inference class
  - ONNX and PyTorch model support
  - Batch inference
  - Result formatting
  - Label decoding
- `load_engine()`: Convenience function to load engine

**Usage**:
```python
# Production inference
from src.nlp.inference import load_engine

engine = load_engine('outputs/model.onnx', model_type='onnx')
result = engine.predict("I love this!")
print(result['emotion'])  # 'joy'

# Batch inference
results = engine.predict_batch(texts, batch_size=32)
```

**Why Inference-Only?**
- Inference doesn't need training utilities (losses, optimizers, etc.)
- Can be deployed independently
- Minimal dependencies for production
- Fast loading and execution

---

### 4. `src/nlp/preprocessing/` - Shared Preprocessing

**Purpose**: Text preprocessing used by both training and inference

**Contents**:
- `normalize.py`: Text cleaning, emoji handling, social media normalization
- `label_mapping.py`: Emotion, style, intent, safety label mappings

**Usage**:
```python
# Both training and inference
from src.nlp.preprocessing import preprocess_text, EMOTION_LABELS

clean_text = preprocess_text(raw_text, lowercase=True)
emotion_id = 5
emotion_name = EMOTION_LABELS[emotion_id]  # 'fear'
```

**Why Shared?**
- Same preprocessing must be applied in both training and inference
- Ensures consistency
- Single source of truth for labels

---

### 5. `src/nlp/safety/` - Shared Safety Tools

**Purpose**: PII detection and scrubbing (optional)

**Contents**:
- `pii_scrub.py`: Presidio integration for privacy

**Usage**:
```python
from src.nlp.safety.pii_scrub import scrub_pii

clean_text = scrub_pii("My email is john@example.com")
# "My email is [EMAIL]"
```

---

## Import Patterns

### Training Scripts

```python
# scripts/train_enhanced.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from training module
from src.nlp.training import (
    set_seed, get_device, MixedPrecisionTrainer,
    FocalLoss, MultiTaskLoss, LossCalculator,
    save_checkpoint, generate_training_report
)

# Import shared models
from src.nlp.models import create_model

# Import shared preprocessing
from src.nlp.preprocessing import preprocess_text
```

### Inference Scripts

```python
# Production inference
from src.nlp.inference import load_engine
from src.nlp.preprocessing import preprocess_text

# Load model
engine = load_engine('model.onnx')

# Preprocess and predict
text = preprocess_text(raw_text)
result = engine.predict(text)
```

### API Server

```python
# src/api/main.py
from src.nlp.inference import EmotionInferenceEngine
from src.nlp.preprocessing import preprocess_text

# Initialize once
engine = EmotionInferenceEngine('model.onnx')

@app.post("/predict")
async def predict(request: Request):
    text = preprocess_text(request.text)
    result = engine.predict(text)
    return result
```

---

## Benefits of This Organization

### 1. **Clean Separation**
- Training code never imported during inference
- Inference code never imported during training
- Shared code clearly identified

### 2. **Minimal Dependencies**
- Inference deployment doesn't need: matplotlib, sklearn, training libraries
- Training doesn't need: ONNX runtime
- Reduces Docker image size and startup time

### 3. **Easy Testing**
```python
# Test training utilities
python src/nlp/training/utils.py

# Test inference engine
python src/nlp/inference/engine.py model.onnx

# Test models (shared)
python src/nlp/models/multi_task_model.py
```

### 4. **Clear Ownership**
- Training team owns `src/nlp/training/`
- Inference/deployment team owns `src/nlp/inference/`
- ML team owns `src/nlp/models/`
- Everyone shares `src/nlp/preprocessing/`

### 5. **Backward Compatibility**
```python
# Old code still works (training/__init__.py imports from models/)
from src.nlp.training import create_model  # Works

# New code uses shared module
from src.nlp.models import create_model  # Preferred
```

---

## Migration Guide

### If You Have Existing Code

**Training Scripts**:
```python
# OLD (still works)
from src.nlp.training.model_architecture import create_model

# NEW (preferred)
from src.nlp.models import create_model
```

**Inference Code**:
```python
# OLD (dummy engine)
from src.nlp.inference import get_engine

# NEW (production engine)
from src.nlp.inference import load_engine
engine = load_engine('model.onnx')
```

---

## Deployment Scenarios

### Scenario 1: Training Pipeline

**Required Modules**:
- `src/nlp/models/` ✓
- `src/nlp/training/` ✓
- `src/nlp/preprocessing/` ✓

**Not Required**:
- `src/nlp/inference/` ✗ (not needed for training)

### Scenario 2: Inference API

**Required Modules**:
- `src/nlp/models/` ✓
- `src/nlp/inference/` ✓
- `src/nlp/preprocessing/` ✓

**Not Required**:
- `src/nlp/training/` ✗ (not needed for inference)

### Scenario 3: Full Development

**Required Modules**:
- `src/nlp/models/` ✓
- `src/nlp/training/` ✓
- `src/nlp/inference/` ✓
- `src/nlp/preprocessing/` ✓

---

## Testing Checklist

After reorganization, verify:

- [ ] Training scripts still work
  ```bash
  python scripts/train_enhanced.py --help
  ```

- [ ] Inference engine loads
  ```bash
  python src/nlp/inference/engine.py outputs/model.onnx onnx
  ```

- [ ] Models import correctly
  ```python
  from src.nlp.models import create_model
  model = create_model()
  ```

- [ ] Training utilities work
  ```bash
  python src/nlp/training/utils.py
  ```

- [ ] API server starts
  ```bash
  python src/api/main.py
  ```

---

## Summary

**Key Principle**: Code is organized by **usage context**, not by type.

- **Training context** → `src/nlp/training/`
- **Inference context** → `src/nlp/inference/`
- **Shared by both** → `src/nlp/models/`, `src/nlp/preprocessing/`

This organization makes it easy to:
- ✅ Deploy inference without training dependencies
- ✅ Train models without inference overhead
- ✅ Share model definitions without duplication
- ✅ Test each component independently
- ✅ Understand what code is used where

---

**Questions?** See individual module docstrings or ask the team!
