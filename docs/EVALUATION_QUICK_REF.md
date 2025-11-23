# Evaluation & Calibration - Quick Reference

## 🎯 Quick Commands

### Calibration & Thresholds
```bash
python scripts/evaluate_calibration.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --metric f1
```
**Outputs**: `reliability_diagram.png`, `thresholds.yaml`, `confusion_matrix.png`, `classification_report.txt`

### Robustness Testing
```bash
python scripts/evaluate_robustness.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --noise-levels 0.1 0.2 0.5
```
**Outputs**: `robustness_curves.png`, `robustness_summary.json`

### Safety Evaluation
```bash
python scripts/evaluate_safety.py --checkpoint outputs/best_model.pt --data data/processed/bootstrap --fn-cost 5.0 --threshold-sweep
```
**Outputs**: `threshold_analysis.png`, `edge_cases.json`, `safety_summary.json`

---

## 📊 Key Metrics

| Metric | Formula | Good Value | Bad Value |
|--------|---------|------------|-----------|
| **ECE** | Avg \|confidence - accuracy\| | < 0.05 ✅ | > 0.20 ❌ |
| **MCE** | Max \|confidence - accuracy\| | < 0.10 ✅ | > 0.30 ❌ |
| **Precision** | TP / (TP + FP) | > 0.85 ✅ | < 0.70 ❌ |
| **Recall** | TP / (TP + FN) | > 0.85 ✅ | < 0.70 ❌ |
| **F1** | 2 × (Prec × Recall) / (Prec + Recall) | > 0.85 ✅ | < 0.70 ❌ |
| **Robustness Drop** | Clean - Noisy @ 20% noise | < 5% ✅ | > 20% ❌ |
| **Safety Cost** | FP + (FN × cost_multiplier) | Lower ✅ | Higher ❌ |

---

## 🔧 Python API - Quick Examples

### Calibration Analysis
```python
from src.nlp.training.evaluation import CalibrationAnalyzer

analyzer = CalibrationAnalyzer(n_bins=10, class_names=['joy', 'sadness', 'anger', 'fear', 'neutral'])
analyzer.add_predictions(y_true, y_probs)

ece = analyzer.compute_ece()  # Expected Calibration Error
mce = analyzer.compute_mce()  # Maximum Calibration Error

analyzer.plot_reliability_diagram('calibration.png')
analyzer.plot_per_class_calibration('per_class.png', num_classes=5)
```

### Threshold Optimization
```python
from src.nlp.training.evaluation import ThresholdOptimizer

optimizer = ThresholdOptimizer(class_names=['joy', 'sadness', 'anger', 'fear', 'neutral'])
thresholds = optimizer.optimize_thresholds(y_true, y_probs, metric='f1')
# {'joy': 0.45, 'sadness': 0.55, 'anger': 0.40, ...}

optimizer.save_thresholds('outputs/thresholds.yaml', thresholds)
loaded = ThresholdOptimizer.load_thresholds('outputs/thresholds.yaml')
```

### Confusion Matrix
```python
from src.nlp.training.evaluation import ConfusionMatrixAnalyzer

analyzer = ConfusionMatrixAnalyzer(class_names=['joy', 'sadness', 'anger', 'fear', 'neutral'])
analyzer.plot_confusion_matrix(y_true, y_pred, 'confusion.png', normalize=True)

confused_pairs = analyzer.find_most_confused_pairs(y_true, y_pred, top_k=5)
# [('sadness', 'neutral', 45), ('fear', 'sadness', 38), ...]

report = analyzer.generate_classification_report(y_true, y_pred)
```

### Noise Injection
```python
from scripts.evaluate_robustness import NoiseInjector

injector = NoiseInjector(noise_level=0.2, seed=42)

# Individual noise types
noisy = injector.add_typos("Hello world")           # → "Helo wrold"
noisy = injector.add_slang("Are you okay?")         # → "R u okay?"
noisy = injector.add_emojis("I'm happy!")           # → "I'm happy! 😊"
noisy = injector.add_repeated_chars("So happy")     # → "Soooo happy"
noisy = injector.add_mixed_language("Thank you")    # → "Gracias"

# Combined noise
noisy = injector.apply_noise("I am very happy", ['typos', 'slang', 'emojis'])
```

### Safety Evaluation
```python
from scripts.evaluate_safety import SafetyEvaluator

evaluator = SafetyEvaluator(fn_cost=5.0)

# Find optimal threshold
threshold, metrics = evaluator.find_optimal_threshold(y_true, y_probs, metric='cost')
# threshold: 0.35
# metrics: {'precision': 0.89, 'recall': 0.92, 'f1': 0.91, 'cost': 25.5, ...}

# Plot analysis
evaluator.plot_threshold_analysis(y_true, y_probs, 'threshold_analysis.png')

# Find edge cases
edge_cases = evaluator.analyze_edge_cases(texts, y_true, y_pred, y_probs)
# {'false_positives': [...], 'false_negatives': [...], 'borderline_cases': [...]}
```

---

## 🎨 Output Files Reference

### Calibration Script
```
reports/calibration/20241123_183930/
├── reliability_diagram.png        # Overall calibration curve
├── per_class_calibration.png      # Per-emotion calibration
├── confusion_matrix.png           # Confusion heatmap
├── classification_report.txt      # Detailed per-class metrics
├── thresholds.yaml               # Optimal thresholds
└── evaluation_summary.json        # All metrics + metadata
```

### Robustness Script
```
reports/robustness/20241123_183930/
├── robustness_curves.png          # Accuracy/F1 vs noise level
└── robustness_summary.json        # Performance degradation data
```

### Safety Script
```
reports/safety/20241123_183930/
├── threshold_analysis.png         # 4-panel threshold sweep
├── edge_cases.json               # FP/FN examples
└── safety_summary.json           # Optimal threshold + metrics
```

---

## 🚀 Common Workflows

### After Training a New Model
```bash
# 1. Quick evaluation
python scripts/evaluate_calibration.py --checkpoint outputs/new_model.pt --data data/processed/bootstrap

# 2. Check calibration
# If ECE > 0.10, consider recalibration

# 3. Update thresholds
# Thresholds automatically saved to outputs/thresholds.yaml

# 4. Test robustness
python scripts/evaluate_robustness.py --checkpoint outputs/new_model.pt --data data/processed/bootstrap

# 5. If deploying safety classifier
python scripts/evaluate_safety.py --checkpoint outputs/new_model.pt --data data/processed/bootstrap --fn-cost 5.0
```

### Debugging Poor Performance
```bash
# 1. Check confusion matrix
python scripts/evaluate_calibration.py --checkpoint outputs/model.pt --data data/processed/bootstrap

# 2. Review most confused pairs in classification_report.txt

# 3. Test robustness to identify weak points
python scripts/evaluate_robustness.py --checkpoint outputs/model.pt --data data/processed/bootstrap --noise-types typos slang

# 4. Review edge cases
python scripts/evaluate_safety.py --checkpoint outputs/model.pt --data data/processed/bootstrap
# Check edge_cases.json for problematic examples
```

### Threshold Tuning
```bash
# Try different optimization metrics
python scripts/evaluate_calibration.py --checkpoint outputs/model.pt --data data/processed/bootstrap --metric f1
python scripts/evaluate_calibration.py --checkpoint outputs/model.pt --data data/processed/bootstrap --metric precision
python scripts/evaluate_calibration.py --checkpoint outputs/model.pt --data data/processed/bootstrap --metric recall

# Compare results in outputs/thresholds.yaml
# Choose based on business requirements:
# - F1: Balanced
# - Precision: Minimize false positives (high confidence predictions)
# - Recall: Minimize false negatives (catch all true cases)
```

---

## 📈 Interpreting Visualizations

### Reliability Diagram
- **Perfect**: Points on diagonal line
- **Overconfident**: Points above diagonal (predicted > actual)
- **Underconfident**: Points below diagonal (predicted < actual)

### Confusion Matrix
- **Bright diagonal**: Good (correct predictions)
- **Bright off-diagonal**: Bad (misclassifications)
- **Look for patterns**: Which classes are confused?

### Robustness Curves
- **Flat lines**: Robust to noise ✅
- **Steep drops**: Sensitive to noise ❌
- **Compare noise types**: Which causes most degradation?

### Threshold Analysis (Safety)
- **Top-left**: Metrics vs threshold (find peak F1)
- **Top-right**: Cost vs threshold (find minimum)
- **Bottom-left**: FP/FN counts (balance trade-off)
- **Bottom-right**: Precision-Recall curve (overall quality)

---

## ⚠️ Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High ECE (> 0.15) | Overconfident predictions | Add label smoothing, temperature scaling |
| All thresholds = 0.5 | Class balance, insufficient data | Check validation set, use stratified split |
| High confusion: A ↔ B | Classes too similar | Add more training data, feature engineering |
| > 20% drop @ 20% noise | Model not robust | Add data augmentation, robust tokenization |
| High safety FN rate | Threshold too high | Lower threshold, add unsafe examples |

---

## 🔗 Links

- **Full Documentation**: `docs/EVALUATION_GUIDE.md`
- **Training Pipeline**: `docs/TRAINING_PIPELINE.md`
- **Model Architecture**: `docs/MODEL_IMPROVEMENTS.md`
- **Code Organization**: `docs/CODE_ORGANIZATION.md`

---

## 💡 Pro Tips

1. **Always run calibration after training** - It's fast and reveals confidence issues
2. **Use cross-validation for thresholds** - Single validation set may be biased
3. **Test robustness on real-world samples** - Synthetic noise != real noise
4. **Monitor metrics over time** - Track calibration drift in production
5. **Review edge cases manually** - Automated metrics miss nuanced errors
6. **Document everything** - Save evaluation reports for every model version

---

**Last Updated**: November 2025  
**Version**: 1.0
