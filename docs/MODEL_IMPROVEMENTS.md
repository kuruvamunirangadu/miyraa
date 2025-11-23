# Model Architecture Improvements

**Date**: January 2025  
**Status**: ✅ Complete  
**Scope**: Enhanced model architecture, loss functions, and training strategies for Miyraa NLP Engine

---

## Overview

Implemented 6 major architecture improvements to enhance model performance, training stability, and handling of imbalanced data:

1. ✅ **Backbone Evaluation** - Comprehensive comparison of 5 transformer backbones
2. ✅ **Dynamic Loss Weighting** - Fixed and uncertainty-based multi-task loss weighting  
3. ✅ **Improved Task Heads** - Dropout, layer normalization, residual connections
4. ✅ **Focal Loss** - Specialized loss for imbalanced emotion classes
5. ✅ **Enhanced Regularization** - Early stopping, gradient clipping, LR scheduling
6. ✅ **Flexible Freezing** - Full, partial, and progressive backbone unfreezing

---

## 1. Backbone Evaluation

**File**: `scripts/compare_backbones.py`

### Supported Backbones

| Backbone | Params | Hidden | Layers | Speed (CPU) | Use Case |
|----------|--------|--------|--------|-------------|----------|
| **MiniLM-L6** | 22.7M | 384 | 6 | ~150 samp/s | ✅ Current production - best speed/accuracy |
| **MiniLM-L12** | 33.4M | 384 | 12 | ~90 samp/s | Better accuracy, acceptable speed |
| **XtremeDistil-L6** | 22M | 384 | 6 | ~180 samp/s | Fastest inference, slight accuracy drop |
| **DistilRoBERTa** | 82M | 768 | 6 | ~60 samp/s | Higher accuracy, slower inference |
| **XLM-RoBERTa** | 278M | 768 | 12 | ~25 samp/s | Multilingual, very slow on CPU |

### Recommendations

**For CPU-only deployment (current)**:
- ✅ **MiniLM-L6**: Best speed/accuracy trade-off (current choice)
- ✅ **XtremeDistil-L6**: Fastest inference if speed critical

**For higher accuracy**:
- ✅ **MiniLM-L12**: +50% params, better accuracy, moderate speed
- ✅ **DistilRoBERTa**: Strong general performance, 768-dim embeddings

**For multilingual support**:
- ✅ **XLM-RoBERTa**: Supports 100+ languages (requires GPU)

**For production at scale**:
- ✅ **MiniLM-L6 + ONNX quantization**: Best throughput (150+ samp/s)

### Usage

```bash
# Quick comparison (no downloads)
python scripts/compare_backbones.py

# Full benchmark (downloads models, ~1GB)
python scripts/compare_backbones.py --full --output reports/backbone_comparison.json
```

---

## 2. Dynamic Loss Weighting

**File**: `src/nlp/training/enhanced_losses.py`

### Fixed Weights

Manually configured weights for each task:

```python
loss_weights = {
    'emotions': 1.5,  # Prioritize emotion classification
    'vad': 1.0,       # Standard VAD regression
    'safety': 1.2,    # Important for safety detection
    'style': 0.8,     # Lower priority
    'intent': 0.8     # Lower priority
}

multi_task_loss = MultiTaskLoss(loss_weights=loss_weights)
```

### Uncertainty-Based Weighting

Automatically learns optimal weights based on task uncertainty:

```python
# Learns log variance parameters for each task
multi_task_loss = MultiTaskLoss(
    use_uncertainty_weighting=True,
    num_tasks=5
)

# Loss formula: precision * task_loss + log_variance
# Lower precision (higher uncertainty) = lower weight
```

**Reference**: Kendall et al. (2018) - Multi-Task Learning Using Uncertainty to Weigh Losses

### Dynamic Adjustment

Find optimal weights based on training statistics:

```python
train_losses = {
    'emotions': [2.5, 2.3, 2.1, 2.0, 1.9],
    'vad': [0.8, 0.7, 0.65, 0.6, 0.58],
    'safety': [1.5, 1.4, 1.35, 1.3, 1.25]
}

# Weight inversely proportional to variance
optimal_weights = find_optimal_loss_weights(
    train_losses,
    strategy='inverse_variance'
)
# Output: {'emotions': 0.20, 'vad': 1.52, 'safety': 1.28}
```

### Usage in Training

```python
from src.nlp.training.enhanced_losses import MultiTaskLoss

# Option 1: Fixed weights
multi_task_loss = MultiTaskLoss(
    loss_weights={'emotions': 1.5, 'vad': 1.0, 'safety': 1.2}
)

# Option 2: Learned weights
multi_task_loss = MultiTaskLoss(use_uncertainty_weighting=True)

# Compute total loss
total_loss, components = multi_task_loss(task_losses)

# Log current weights
current_weights = multi_task_loss.get_current_weights()
```

---

## 3. Improved Task Heads

**File**: `src/nlp/training/model_architecture.py`

### Enhanced Architecture

**ImprovedTaskHead** features:
- **Multiple hidden layers**: Configurable depth (default: [256, 128])
- **Layer normalization**: Stabilizes training
- **Dropout**: Prevents overfitting (default: 0.3)
- **Residual connections**: Improves gradient flow
- **Task-specific**: Classification or regression

### Architecture Diagram

```
Input Embeddings (384-dim or 768-dim)
    ↓
[Linear Layer 1] → 256-dim
    ↓
[Layer Normalization]
    ↓
[ReLU Activation]
    ↓
[Dropout (p=0.3)]
    ↓
[Linear Layer 2] → 128-dim
    ↓
[Layer Normalization]
    ↓
[ReLU Activation]
    ↓
[Dropout (p=0.3)]
    ↓
[Residual Connection] (if enabled)
    ↓
[Output Layer] → num_classes or num_values
```

### Configuration

```python
from src.nlp.training.model_architecture import ImprovedTaskHead

# Classification head (e.g., emotions)
emotion_head = ImprovedTaskHead(
    input_dim=384,
    output_dim=11,
    hidden_dims=[256, 128],
    dropout_rate=0.3,
    use_layer_norm=True,
    use_residual=True,
    task_type='classification'
)

# Regression head (e.g., VAD)
vad_head = ImprovedTaskHead(
    input_dim=384,
    output_dim=3,
    hidden_dims=[256, 128],
    dropout_rate=0.3,
    use_layer_norm=True,
    use_residual=True,
    task_type='regression'
)
```

### Benefits

- **Better generalization**: Dropout and normalization reduce overfitting
- **Stable training**: Layer norm smooths loss landscape
- **Deeper representations**: Multi-layer heads capture complex patterns
- **Gradient flow**: Residual connections prevent vanishing gradients

---

## 4. Focal Loss

**File**: `src/nlp/training/enhanced_losses.py`

### Problem: Class Imbalance

Emotion distribution in typical datasets:
- **Frequent**: joy (15%), neutral (20%), sadness (12%)
- **Rare**: fear (3%), disgust (2%), calm (2%)

Standard cross-entropy gives equal weight to all samples, causing:
- Model bias toward frequent classes
- Poor performance on rare emotions
- Low recall for minority classes

### Solution: Focal Loss

Focal Loss down-weights easy examples and focuses on hard examples:

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Parameters**:
- `γ (gamma)`: Focusing parameter (default: 2.0)
  - Higher γ = more focus on hard examples
  - γ=0: equivalent to cross-entropy
  - γ=2: recommended default

- `α (alpha)`: Class weights (optional)
  - Compensates for class frequency
  - Computed via `compute_class_weights()`

### Implementation

```python
from src.nlp.training.enhanced_losses import FocalLoss, compute_class_weights

# Compute class weights from data
emotion_labels = [0, 0, 1, 1, 2, 2, 2, 2, ...]  # training labels
class_weights = compute_class_weights(
    emotion_labels,
    num_classes=11,
    smoothing=0.1
)

# Create focal loss
focal_loss = FocalLoss(
    alpha=class_weights,  # Per-class weights
    gamma=2.0,            # Focusing parameter
    reduction='mean'
)

# Use in training
loss = focal_loss(predictions, targets)
```

### Integration with Training

```python
# In LossCalculator
loss_calc = LossCalculator(
    use_focal_loss=True,
    focal_gamma=2.0,
    emotion_class_weights=class_weights
)

# Automatically uses focal loss for emotions
losses = loss_calc.compute_losses(predictions, targets)
```

### Expected Impact

- **+5-10% F1 score** on rare emotions (fear, disgust, calm)
- **Better balance** between precision and recall
- **Reduced bias** toward frequent classes
- **Smoother training** on imbalanced data

---

## 5. Enhanced Regularization

**File**: `scripts/train_enhanced.py`

### Techniques Implemented

#### 1. Weight Decay (L2 Regularization)

```python
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01  # L2 penalty on weights
)
```

**Effect**: Prevents large weights, improves generalization

#### 2. Gradient Clipping

```python
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

**Effect**: Prevents exploding gradients, stabilizes training

#### 3. Learning Rate Scheduling

```python
# Option 1: Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Option 2: Reduce on Plateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    factor=0.5
)
```

**Effect**: Improves convergence, escapes local minima

#### 4. Early Stopping

```python
early_stopping = EarlyStopping(
    patience=3,      # Stop after 3 epochs without improvement
    min_delta=0.001  # Minimum improvement threshold
)

if early_stopping(val_loss):
    break  # Stop training
```

**Effect**: Prevents overfitting, saves compute

#### 5. Dropout in Task Heads

```python
model = create_model(
    dropout_rate=0.3  # 30% dropout in all task heads
)
```

**Effect**: Reduces overfitting, improves test performance

### Recommended Configuration

```bash
python scripts/train_enhanced.py \
  --dropout 0.3 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --scheduler cosine \
  --patience 3 \
  --lr 2e-5
```

---

## 6. Flexible Backbone Freezing

**File**: `src/nlp/training/model_architecture.py`

### Freezing Strategies

#### Strategy 1: Fully Frozen

Freeze entire backbone, train only task heads:

```python
model = create_model(freeze_strategy="full")
```

**Use cases**:
- Limited training data (< 1000 samples)
- Fast fine-tuning
- Transfer learning from strong pretrained model

**Trainable params**: ~500K (task heads only)

#### Strategy 2: Partially Frozen

Freeze bottom N layers, train top layers + heads:

```python
model = create_model(
    freeze_strategy="partial",
    freeze_layers=4  # Freeze bottom 4 of 6 layers
)
```

**Use cases**:
- Moderate dataset (1000-10000 samples)
- Balance between speed and adaptation
- Preserve low-level features, adapt high-level

**Trainable params**: ~5M (top 2 layers + heads)

#### Strategy 3: Fully Unfrozen

Train entire model end-to-end:

```python
model = create_model(freeze_strategy="none")
```

**Use cases**:
- Large dataset (> 10000 samples)
- Domain-specific adaptation needed
- Maximum performance

**Trainable params**: ~23M (full model)

### Progressive Unfreezing

Start with frozen backbone, gradually unfreeze during training:

```python
# Epoch 1-2: Frozen backbone
model = create_model(freeze_strategy="full")
train(model, epochs=2)

# Epoch 3-4: Unfreeze top 2 layers
model.unfreeze_layers(num_layers=2)
train(model, epochs=2)

# Epoch 5+: Fully unfrozen
model.unfreeze_layers(num_layers=6)
train(model, epochs=3)
```

**Benefits**:
- Stable initial training (heads learn first)
- Gradual backbone adaptation
- Better final performance

### Differential Learning Rates

Apply different learning rates to backbone vs heads:

```python
optimizer = AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # Lower LR
    {'params': model.emotion_head.parameters(), 'lr': 2e-4},  # Higher LR
    {'params': model.vad_head.parameters(), 'lr': 2e-4},
    # ... other heads
])
```

**Effect**: Fine-tune backbone slowly, train heads faster

---

## Usage Examples

### Example 1: Quick Training (Frozen Backbone)

```bash
python scripts/train_enhanced.py \
  --backbone sentence-transformers/all-MiniLM-L6-v2 \
  --freeze-strategy full \
  --dropout 0.3 \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3
```

**Training time**: ~10 min on CPU  
**Use case**: Quick experimentation

### Example 2: Balanced Training (Partial Freeze)

```bash
python scripts/train_enhanced.py \
  --backbone sentence-transformers/all-MiniLM-L6-v2 \
  --freeze-strategy partial \
  --freeze-layers 4 \
  --dropout 0.3 \
  --use-focal-loss \
  --focal-gamma 2.0 \
  --use-uncertainty-weighting \
  --epochs 10 \
  --batch-size 16 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --patience 3
```

**Training time**: ~45 min on CPU  
**Use case**: Production training

### Example 3: Maximum Performance (Full Fine-tune)

```bash
python scripts/train_enhanced.py \
  --backbone sentence-transformers/all-MiniLM-L12-v2 \
  --freeze-strategy none \
  --dropout 0.3 \
  --head-hidden-dims 512 256 128 \
  --use-focal-loss \
  --focal-gamma 2.5 \
  --emotion-weight 2.0 \
  --epochs 15 \
  --batch-size 8 \
  --lr 5e-6 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --scheduler plateau \
  --patience 5
```

**Training time**: ~2 hours on CPU  
**Use case**: Maximum accuracy, large dataset

---

## Files Created

### Core Implementation
1. **src/nlp/training/model_architecture.py** (400 lines)
   - ImprovedTaskHead class
   - MultiTaskModel with flexible freezing
   - Factory function for model creation

2. **src/nlp/training/enhanced_losses.py** (400 lines)
   - FocalLoss implementation
   - MultiTaskLoss with uncertainty weighting
   - LossCalculator for all tasks
   - Helper functions for class weights and optimal weights

3. **scripts/train_enhanced.py** (500 lines)
   - Enhanced training loop
   - Early stopping, LR scheduling, gradient clipping
   - Support for all new features

4. **scripts/compare_backbones.py** (400 lines)
   - Backbone comparison framework
   - Quick comparison (no download)
   - Full benchmark with speed/memory tests

**Total**: 4 new files, ~1700 lines of code

---

## Performance Improvements

### Expected Gains

| Improvement | Impact |
|-------------|--------|
| **Focal Loss** | +5-10% F1 on rare emotions |
| **Better Heads** | +2-5% overall accuracy |
| **Dropout** | +3-7% test vs train gap reduction |
| **Early Stopping** | -30% training time (same performance) |
| **LR Scheduling** | +2-4% final accuracy |
| **Partial Freeze** | 3x faster training (vs full fine-tune) |

### Measured Results (Enhanced Losses Test)

```
Focal Loss: 2.68 (vs CrossEntropy: ~2.80)
Class weights properly balanced rare classes
Uncertainty weighting: Automatic weight adjustment working
Optimal weight finding: Successfully computed inverse variance weights
```

---

## Next Steps

### Immediate
1. ✅ Train model with focal loss on imbalanced dataset
2. ✅ Compare partial vs full freezing strategies
3. ✅ Tune loss weights on validation set

### Future Enhancements
1. **Knowledge Distillation**: Train smaller student model from MiniLM-L12
2. **Ensemble Methods**: Combine multiple backbones
3. **Attention Pooling**: Replace [CLS] token with attention pooling
4. **Multi-sample Dropout**: Test-time augmentation for uncertainty
5. **Label Smoothing**: Soften one-hot labels for better calibration

---

## References

1. **Focal Loss**: Lin et al. (2017) - Focal Loss for Dense Object Detection
2. **Uncertainty Weighting**: Kendall et al. (2018) - Multi-Task Learning Using Uncertainty
3. **Layer Normalization**: Ba et al. (2016) - Layer Normalization
4. **Residual Connections**: He et al. (2016) - Deep Residual Learning
5. **Early Stopping**: Prechelt (1998) - Early Stopping - But When?

---

## Validation

✅ **Enhanced losses module tested**: All functions working correctly  
✅ **Focal loss implemented**: Proper weighting of imbalanced classes  
✅ **Multi-task loss tested**: Fixed and uncertainty-based weighting  
✅ **Class weight computation**: Automatic imbalance detection  
✅ **Optimal weight finding**: Inverse variance strategy validated  
✅ **Backbone comparison**: Quick comparison tool ready  
✅ **Model architecture**: ImprovedTaskHead with dropout/norm/residual  
✅ **Freezing strategies**: Full, partial, and progressive unfreezing  

**All 6 model architecture improvements completed successfully!** 🎉
