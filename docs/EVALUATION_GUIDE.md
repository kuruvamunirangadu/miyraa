# Evaluation & Calibration Guide

Complete guide to model evaluation, calibration, and robustness testing for the Miyraa NLP Emotion Engine.

## Table of Contents

1. [Overview](#overview)
2. [Calibration Analysis](#calibration-analysis)
3. [Threshold Optimization](#threshold-optimization)
4. [Confusion Matrix Analysis](#confusion-matrix-analysis)
5. [Noise Robustness Testing](#noise-robustness-testing)
6. [Safety Classifier Evaluation](#safety-classifier-evaluation)
7. [Complete Workflow](#complete-workflow)
8. [API Reference](#api-reference)

---

## Overview

The evaluation suite provides comprehensive model quality assessment:

- **Calibration**: Measures how well predicted probabilities match actual outcomes
- **Thresholds**: Optimizes per-class decision thresholds for F1/precision/recall
- **Confusion Matrix**: Identifies systematic classification errors
- **Robustness**: Tests resilience to typos, slang, emojis, and other noise
- **Safety Analysis**: Minimizes false positives and false negatives in content moderation

### Quick Start

```bash
# 1. Calibration and thresholds
python scripts/evaluate_calibration.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --metric f1

# 2. Noise robustness
python scripts/evaluate_robustness.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --noise-levels 0.1 0.2 0.5

# 3. Safety evaluation
python scripts/evaluate_safety.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --fn-cost 5.0 \
    --threshold-sweep
```

---

## Calibration Analysis

### What is Calibration?

A well-calibrated model's predicted probabilities match actual frequencies:
- If model predicts 70% confidence, it should be correct 70% of the time
- Calibration is crucial for decision-making based on model confidence

### Metrics

#### Expected Calibration Error (ECE)
- Average absolute difference between confidence and accuracy across bins
- **Lower is better** (0 = perfect calibration)
- Formula: `ECE = Σ |accuracy(bin) - confidence(bin)| × P(bin)`

#### Maximum Calibration Error (MCE)
- Maximum calibration error across any bin
- **Lower is better**
- More sensitive to worst-case calibration issues

### Usage

```python
from src.nlp.training.evaluation import CalibrationAnalyzer

# Initialize
analyzer = CalibrationAnalyzer(n_bins=10, class_names=['joy', 'sadness', 'anger', 'fear', 'neutral'])

# Add predictions
analyzer.add_predictions(y_true, y_probs)

# Compute metrics
ece = analyzer.compute_ece()
mce = analyzer.compute_mce()

print(f"ECE: {ece:.4f}")  # e.g., 0.0234
print(f"MCE: {mce:.4f}")  # e.g., 0.0512

# Generate plots
analyzer.plot_reliability_diagram('calibration.png')
analyzer.plot_per_class_calibration('per_class_calibration.png', num_classes=5)
```

### Command Line

```bash
python scripts/evaluate_calibration.py \
    --checkpoint outputs/best_model_20241123_183930.pt \
    --data data/processed/bootstrap \
    --output reports/calibration \
    --batch-size 32
```

**Outputs:**
- `reliability_diagram.png`: Overall calibration curve
- `per_class_calibration.png`: Per-emotion calibration
- `evaluation_summary.json`: Metrics and metadata

### Interpreting Results

**ECE < 0.05**: Excellent calibration ✅
**ECE 0.05-0.10**: Good calibration ✓
**ECE 0.10-0.20**: Fair calibration ⚠
**ECE > 0.20**: Poor calibration ❌ (consider recalibration techniques)

**Reliability Diagram:**
- Points near diagonal = well calibrated
- Points above diagonal = overconfident (too high probability)
- Points below diagonal = underconfident (too low probability)

---

## Threshold Optimization

### Why Optimize Thresholds?

Default 0.5 threshold is often suboptimal. Each emotion class has different:
- Class balance (some emotions rarer than others)
- Cost of false positives vs false negatives
- Desired precision/recall trade-off

### Usage

```python
from src.nlp.training.evaluation import ThresholdOptimizer

# Initialize
optimizer = ThresholdOptimizer(class_names=['joy', 'sadness', 'anger', 'fear', 'neutral'])

# Find optimal thresholds
thresholds = optimizer.optimize_thresholds(
    y_true, 
    y_probs, 
    metric='f1',  # or 'precision', 'recall'
    num_classes=5
)

print(thresholds)
# {'joy': 0.45, 'sadness': 0.55, 'anger': 0.40, 'fear': 0.50, 'neutral': 0.35}

# Save thresholds
optimizer.save_thresholds(
    'outputs/thresholds.yaml',
    thresholds,
    metadata={'metric': 'f1', 'n_samples': 1000}
)

# Load thresholds for inference
thresholds = ThresholdOptimizer.load_thresholds('outputs/thresholds.yaml')
```

### Command Line

```bash
# Optimize for F1 (default)
python scripts/evaluate_calibration.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --metric f1

# Optimize for precision (fewer false positives)
python scripts/evaluate_calibration.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --metric precision

# Optimize for recall (fewer false negatives)
python scripts/evaluate_calibration.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --metric recall
```

**Output:** `outputs/thresholds.yaml`

```yaml
thresholds:
  joy: 0.45
  sadness: 0.55
  anger: 0.40
  fear: 0.50
  neutral: 0.35
metadata:
  metric: f1
  n_samples: 1000
  ece: 0.0234
  mce: 0.0512
  timestamp: '20241123_183930'
```

### Using Optimized Thresholds in Inference

```python
from src.nlp.inference.engine import EmotionInferenceEngine
from src.nlp.training.evaluation import ThresholdOptimizer

# Load thresholds
thresholds = ThresholdOptimizer.load_thresholds('outputs/thresholds.yaml')

# Apply during inference
engine = EmotionInferenceEngine('outputs/best_model.pt')
result = engine.predict("I'm so happy today!")

# Apply custom threshold
emotion_probs = result['emotion']['probabilities']
emotion_labels = result['emotion']['labels']

for label, prob in zip(emotion_labels, emotion_probs):
    threshold = thresholds.get(label, 0.5)
    if prob >= threshold:
        print(f"✓ {label}: {prob:.3f} (threshold: {threshold:.3f})")
```

---

## Confusion Matrix Analysis

### Overview

Confusion matrices reveal:
- Which emotion pairs are most confused
- Per-class precision, recall, F1
- Support (samples per class)
- Systematic biases

### Usage

```python
from src.nlp.training.evaluation import ConfusionMatrixAnalyzer

# Initialize
analyzer = ConfusionMatrixAnalyzer(class_names=['joy', 'sadness', 'anger', 'fear', 'neutral'])

# Plot confusion matrix
analyzer.plot_confusion_matrix(
    y_true, 
    y_pred,
    'confusion_matrix.png',
    normalize=True  # Show percentages
)

# Find most confused pairs
confused_pairs = analyzer.find_most_confused_pairs(y_true, y_pred, top_k=5)
for true_class, pred_class, count in confused_pairs:
    print(f"{true_class} → {pred_class}: {count} times")

# Output:
# sadness → neutral: 45 times
# fear → sadness: 38 times
# anger → fear: 32 times
# neutral → joy: 28 times
# joy → neutral: 25 times

# Generate classification report
report = analyzer.generate_classification_report(
    y_true, 
    y_pred,
    output_path='classification_report.txt'
)
print(report)
```

### Command Line

```bash
python scripts/evaluate_calibration.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap
```

**Outputs:**
- `confusion_matrix.png`: Heatmap visualization
- `classification_report.txt`: Detailed per-class metrics

### Interpreting Confusion Matrix

**Diagonal values:** Correct predictions (should be high)
**Off-diagonal values:** Misclassifications (should be low)

**Common patterns:**
- **sadness ↔ neutral**: Subtle emotional signals
- **anger ↔ fear**: Both negative, high arousal
- **joy ↔ neutral**: Positive but low intensity
- **fear ↔ sadness**: Both negative, similar valence

**Classification Report Example:**

```
              precision    recall  f1-score   support

         joy     0.8542    0.8231    0.8384       195
     sadness     0.7891    0.8156    0.8021       180
       anger     0.8234    0.7891    0.8059       165
        fear     0.7645    0.7912    0.7776       172
     neutral     0.8123    0.8345    0.8233       188

    accuracy                         0.8107       900
   macro avg     0.8087    0.8107    0.8095       900
weighted avg     0.8113    0.8107    0.8109       900
```

**Key metrics:**
- **Precision**: Of predicted X, what % were actually X?
- **Recall**: Of actual X, what % were predicted as X?
- **F1**: Harmonic mean of precision and recall

---

## Noise Robustness Testing

### Overview

Real-world text is messy. Test model resilience to:

1. **Typos**: Character swaps, deletions, insertions, keyboard neighbors
2. **Slang**: "you" → "u", "tonight" → "2nite", "thanks" → "thx"
3. **Repeated chars**: "sooo happy", "yessss"
4. **Emojis**: 😀 😂 😢 😡 👍
5. **Mixed languages**: "gracias for the help", "that was tres bien"
6. **Special chars**: "Really!!!", "What???"

### Usage

```python
from scripts.evaluate_robustness import NoiseInjector

# Initialize
injector = NoiseInjector(noise_level=0.2, seed=42)

# Apply individual noise types
noisy_text = injector.add_typos("Hello world")  # → "Helo wrold"
noisy_text = injector.add_slang("Are you okay?")  # → "R u okay?"
noisy_text = injector.add_emojis("I'm happy!")  # → "I'm happy! 😊"

# Apply multiple noise types
noisy_text = injector.apply_noise(
    "I am very happy today",
    noise_types=['typos', 'slang', 'emojis']
)
# → "I am vry happy 2day 😀"
```

### Command Line

```bash
# Test all noise types
python scripts/evaluate_robustness.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --noise-levels 0.1 0.2 0.5 \
    --n-samples 500

# Test specific noise types
python scripts/evaluate_robustness.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --noise-types typos slang \
    --noise-levels 0.2 0.5
```

**Outputs:**
- `robustness_curves.png`: Accuracy/F1 vs noise level for each noise type
- `robustness_summary.json`: Detailed performance degradation metrics

### Interpreting Results

**Performance drops:**
- **< 5% drop at 20% noise**: Excellent robustness ✅
- **5-10% drop at 20% noise**: Good robustness ✓
- **10-20% drop at 20% noise**: Fair robustness ⚠
- **> 20% drop at 20% noise**: Poor robustness ❌

**Noise type sensitivity:**
- **Typos**: Most realistic, should be most robust
- **Slang**: Common in social media, important for real-world deployment
- **Emojis**: Context-dependent, can improve or confuse
- **Mixed languages**: Critical for multilingual users

**Example summary:**

```json
{
  "clean_performance": {
    "accuracy": 0.8234,
    "f1": 0.8156
  },
  "typos": {
    "0.1": {"accuracy": 0.8123, "accuracy_drop": 0.0111, "f1": 0.8045, "f1_drop": 0.0111},
    "0.2": {"accuracy": 0.7891, "accuracy_drop": 0.0343, "f1": 0.7812, "f1_drop": 0.0344},
    "0.5": {"accuracy": 0.7234, "accuracy_drop": 0.1000, "f1": 0.7156, "f1_drop": 0.1000}
  }
}
```

**Recommendations based on results:**
- High drop on typos → Add data augmentation with typos
- High drop on slang → Include informal text in training data
- High drop on emojis → Add emoji preprocessing or train on emoji-rich data
- High drop on mixed languages → Use multilingual backbone (e.g., XLM-R)

---

## Safety Classifier Evaluation

### Overview

Safety classification has asymmetric costs:
- **False Positive (FP)**: Safe content flagged as unsafe → user frustration
- **False Negative (FN)**: Unsafe content passes through → platform harm

**FN is typically 3-10x more costly than FP**

### Cost-Weighted Metrics

```
Total Cost = FP_count + (FN_count × FN_cost_multiplier)
```

Example with FN cost = 5:
- 10 FP + 2 FN = 10 + (2 × 5) = 20 cost
- 5 FP + 4 FN = 5 + (4 × 5) = 25 cost (worse, despite fewer total errors)

### Usage

```python
from scripts.evaluate_safety import SafetyEvaluator

# Initialize with FN cost
evaluator = SafetyEvaluator(fn_cost=5.0)

# Find optimal threshold
optimal_threshold, metrics = evaluator.find_optimal_threshold(
    y_true, 
    y_probs, 
    metric='cost'
)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"False Positives: {metrics['fp_count']}")
print(f"False Negatives: {metrics['fn_count']}")
print(f"Weighted Cost: {metrics['cost']:.2f}")

# Plot threshold analysis
evaluator.plot_threshold_analysis(
    y_true, 
    y_probs,
    'threshold_analysis.png'
)

# Analyze edge cases
edge_cases = evaluator.analyze_edge_cases(
    texts, y_true, y_pred, y_probs, num_examples=10
)

# Review problematic examples
for case in edge_cases['false_negatives']:
    print(f"MISSED UNSAFE: {case['text']}")
    print(f"  Confidence: {case['confidence']:.3f}\n")
```

### Command Line

```bash
# Standard evaluation
python scripts/evaluate_safety.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --fn-cost 5.0

# With threshold sweep
python scripts/evaluate_safety.py \
    --checkpoint outputs/best_model.pt \
    --data data/processed/bootstrap \
    --fn-cost 5.0 \
    --threshold-sweep
```

**Outputs:**
- `threshold_analysis.png`: 4-panel visualization
  - Metrics (precision, recall, F1) vs threshold
  - Cost-weighted score vs threshold
  - FP vs FN counts
  - Precision-Recall curve
- `edge_cases.json`: Problematic examples
- `safety_summary.json`: Metrics and recommendations

### Interpreting Results

**Threshold recommendations:**
- **High FN rate**: Lower threshold (catch more unsafe content, accept more FP)
- **High FP rate**: Raise threshold (reduce false alarms, risk missing unsafe content)
- **Balanced**: Current threshold is appropriate

**Edge cases to review:**

1. **False Positives (safe flagged as unsafe)**
   - Sarcasm: "Oh great, another wonderful day" (sarcastic, but safe)
   - Context-dependent: "kill it!" (gaming context, not threat)
   - Medical/educational: Clinical descriptions of harm

2. **False Negatives (unsafe missed)**
   - Subtle threats: "It would be a shame if..."
   - Coded language: Slang or euphemisms for harmful content
   - Context-switching: Safe followed by unsafe in same text

**Example threshold analysis:**

```
Optimal threshold: 0.35
  Precision: 0.8945
  Recall: 0.9234
  F1: 0.9087
  Cost: 25.5
  False Positives: 15
  False Negatives: 2

Recommendation:
  ✓ Threshold appears well-balanced.
  FN rate is low (2), acceptable given FP count.
```

---

## Complete Workflow

### Full Evaluation Pipeline

```bash
#!/bin/bash
# evaluate_all.sh - Complete evaluation pipeline

CHECKPOINT="outputs/best_model_20241123_183930.pt"
DATA="data/processed/bootstrap"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="reports/full_evaluation_${TIMESTAMP}"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "FULL MODEL EVALUATION PIPELINE"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATA"
echo "Output: $OUTPUT_DIR"
echo ""

# 1. Calibration and thresholds
echo "1. Running calibration analysis..."
python scripts/evaluate_calibration.py \
    --checkpoint $CHECKPOINT \
    --data $DATA \
    --output $OUTPUT_DIR/calibration \
    --metric f1 \
    --batch-size 32

# 2. Noise robustness
echo ""
echo "2. Running robustness testing..."
python scripts/evaluate_robustness.py \
    --checkpoint $CHECKPOINT \
    --data $DATA \
    --output $OUTPUT_DIR/robustness \
    --noise-levels 0.1 0.2 0.5 \
    --n-samples 500 \
    --batch-size 32

# 3. Safety evaluation
echo ""
echo "3. Running safety evaluation..."
python scripts/evaluate_safety.py \
    --checkpoint $CHECKPOINT \
    --data $DATA \
    --output $OUTPUT_DIR/safety \
    --fn-cost 5.0 \
    --threshold-sweep \
    --batch-size 32

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated reports:"
echo "  - Calibration: $OUTPUT_DIR/calibration/"
echo "  - Robustness: $OUTPUT_DIR/robustness/"
echo "  - Safety: $OUTPUT_DIR/safety/"
```

### Python API Workflow

```python
from src.nlp.training.evaluation import (
    CalibrationAnalyzer,
    ThresholdOptimizer,
    ConfusionMatrixAnalyzer
)
from scripts.evaluate_robustness import NoiseInjector, evaluate_on_noisy_data
from scripts.evaluate_safety import SafetyEvaluator

# Load model and data
model, tokenizer = load_model_and_tokenizer('outputs/best_model.pt', device)
y_true, y_probs, y_pred = collect_predictions(model, dataloader, device)

# 1. Calibration
calibration = CalibrationAnalyzer(n_bins=10, class_names=class_names)
calibration.add_predictions(y_true, y_probs)
ece = calibration.compute_ece()
calibration.plot_reliability_diagram('calibration.png')

# 2. Thresholds
optimizer = ThresholdOptimizer(class_names=class_names)
thresholds = optimizer.optimize_thresholds(y_true, y_probs, metric='f1')
optimizer.save_thresholds('outputs/thresholds.yaml', thresholds)

# 3. Confusion matrix
cm_analyzer = ConfusionMatrixAnalyzer(class_names=class_names)
cm_analyzer.plot_confusion_matrix(y_true, y_pred, 'confusion.png')
report = cm_analyzer.generate_classification_report(y_true, y_pred)

# 4. Robustness
injector = NoiseInjector(noise_level=0.2)
noisy_texts = [injector.add_typos(text) for text in texts]
noisy_acc, noisy_f1 = evaluate_on_noisy_data(model, tokenizer, noisy_texts, y_true, device)

# 5. Safety
safety_eval = SafetyEvaluator(fn_cost=5.0)
safety_threshold, safety_metrics = safety_eval.find_optimal_threshold(
    y_true_safety, y_probs_safety, metric='cost'
)

print(f"Calibration ECE: {ece:.4f}")
print(f"Optimal thresholds: {thresholds}")
print(f"Noisy text accuracy: {noisy_acc:.4f}")
print(f"Safety threshold: {safety_threshold:.3f}")
```

---

## API Reference

### CalibrationAnalyzer

```python
class CalibrationAnalyzer:
    def __init__(self, n_bins: int = 10, class_names: Optional[List[str]] = None)
    
    def add_predictions(self, y_true: np.ndarray, y_probs: np.ndarray) -> None
    
    def compute_ece(self) -> float
    
    def compute_mce(self) -> float
    
    def plot_reliability_diagram(self, output_path: str, title: str = "Reliability Diagram") -> None
    
    def plot_per_class_calibration(self, output_path: str, num_classes: int) -> None
```

### ThresholdOptimizer

```python
class ThresholdOptimizer:
    def __init__(self, class_names: Optional[List[str]] = None)
    
    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        metric: str = 'f1',
        num_classes: Optional[int] = None
    ) -> Dict[str, float]
    
    def save_thresholds(
        self,
        output_path: str,
        thresholds: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> None
    
    @staticmethod
    def load_thresholds(thresholds_path: str) -> Dict[str, float]
```

### ConfusionMatrixAnalyzer

```python
class ConfusionMatrixAnalyzer:
    def __init__(self, class_names: List[str])
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        normalize: bool = True
    ) -> None
    
    def find_most_confused_pairs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, str, int]]
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Optional[str] = None
    ) -> str
```

### NoiseInjector

```python
class NoiseInjector:
    def __init__(self, noise_level: float = 0.2, seed: int = 42)
    
    def add_typos(self, text: str) -> str
    
    def add_slang(self, text: str) -> str
    
    def add_repeated_chars(self, text: str) -> str
    
    def add_emojis(self, text: str) -> str
    
    def add_mixed_language(self, text: str) -> str
    
    def add_special_chars(self, text: str) -> str
    
    def apply_noise(self, text: str, noise_types: List[str]) -> str
```

### SafetyEvaluator

```python
class SafetyEvaluator:
    def __init__(self, fn_cost: float = 3.0)
    
    def compute_cost_weighted_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        metric: str = 'cost'
    ) -> Tuple[float, Dict]
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        output_path: str
    ) -> None
    
    def analyze_edge_cases(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        num_examples: int = 10
    ) -> Dict
```

---

## Best Practices

### 1. Regular Calibration Checks
- Run calibration analysis after every major model update
- Track ECE/MCE over time to detect calibration drift
- Recalibrate if ECE > 0.10

### 2. Dynamic Threshold Adjustment
- Re-optimize thresholds when:
  - Training data distribution changes
  - Deployment environment changes
  - Business requirements change (e.g., prioritize precision over recall)
- Keep threshold history for A/B testing

### 3. Continuous Robustness Monitoring
- Test on real-world noisy samples periodically
- Add failing examples to training data
- Track robustness metrics in production

### 4. Safety-First Approach
- Always evaluate safety classifier separately
- Monitor FN rate in production (critical metric)
- Set up alerts for FN rate increases
- Review edge cases manually

### 5. Documentation
- Document all evaluation results
- Track metrics over model versions
- Share reports with stakeholders
- Include evaluation results in model cards

---

## Troubleshooting

### High ECE (Poor Calibration)

**Causes:**
- Model overfitting
- Class imbalance
- Focal loss without temperature scaling

**Solutions:**
- Add label smoothing
- Apply temperature scaling
- Use calibration techniques (Platt scaling, isotonic regression)
- Increase training data

### Suboptimal Thresholds

**Causes:**
- Strong class imbalance
- Evaluation metric doesn't match business goal
- Insufficient validation data

**Solutions:**
- Use stratified sampling
- Align metric with business objective
- Increase validation set size
- Use cross-validation for threshold selection

### Poor Robustness

**Causes:**
- Training data too clean
- Model overfits to spelling/grammar
- Lack of augmentation

**Solutions:**
- Add noise augmentation to training
- Use robust tokenization
- Train on diverse data sources (social media, forums)
- Use character-level or subword models

### High Safety FN Rate

**Causes:**
- Threshold too high
- Insufficient unsafe examples in training
- Subtle/coded language not represented

**Solutions:**
- Lower threshold (accept more FP)
- Collect more unsafe examples
- Add adversarial examples
- Use ensemble methods

---

## Next Steps

1. **Implement Calibration Fixes**: If ECE > 0.10, apply temperature scaling
2. **Deploy Thresholds**: Update inference engine with optimized thresholds
3. **Monitor Production**: Track metrics in real-world deployment
4. **Iterate**: Continuously improve based on evaluation insights

For implementation details, see:
- `src/nlp/training/evaluation.py` - Core evaluation classes
- `scripts/evaluate_calibration.py` - Calibration script
- `scripts/evaluate_robustness.py` - Robustness script
- `scripts/evaluate_safety.py` - Safety script

---

**Author**: Miyraa Team  
**Date**: November 2025  
**Version**: 1.0
