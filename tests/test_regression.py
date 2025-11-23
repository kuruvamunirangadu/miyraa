"""
Regression tests for accuracy monitoring.
Tests model accuracy on mini validation set and alerts on degradation.
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


# Sample validation data structure
VALIDATION_DATA = [
    {"text": "I am so happy today!", "emotion": "joy", "sentiment": "positive"},
    {"text": "This makes me angry", "emotion": "anger", "sentiment": "negative"},
    {"text": "I feel sad and lonely", "emotion": "sadness", "sentiment": "negative"},
    {"text": "This is scary", "emotion": "fear", "sentiment": "negative"},
    {"text": "What a wonderful surprise!", "emotion": "joy", "sentiment": "positive"},
    {"text": "I'm feeling neutral about this", "emotion": "neutral", "sentiment": "neutral"},
    {"text": "This is disgusting", "emotion": "disgust", "sentiment": "negative"},
    {"text": "I'm so excited!", "emotion": "joy", "sentiment": "positive"},
    {"text": "This is frustrating", "emotion": "anger", "sentiment": "negative"},
    {"text": "I'm worried about this", "emotion": "fear", "sentiment": "negative"},
]


# Baseline metrics (to be updated when model improves)
BASELINE_METRICS = {
    "emotion": {
        "accuracy": 0.85,
        "f1_macro": 0.83,
        "per_class_f1": {
            "joy": 0.88,
            "anger": 0.82,
            "sadness": 0.80,
            "fear": 0.78,
            "disgust": 0.75,
            "neutral": 0.85
        }
    },
    "sentiment": {
        "accuracy": 0.90,
        "f1_macro": 0.89,
        "per_class_f1": {
            "positive": 0.91,
            "negative": 0.88,
            "neutral": 0.88
        }
    }
}


# Allowed degradation threshold (2%)
DEGRADATION_THRESHOLD = 0.02


def load_baseline_metrics() -> Dict:
    """Load baseline metrics from file or use defaults"""
    metrics_path = Path("reports/baseline_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return BASELINE_METRICS


def save_current_metrics(metrics: Dict):
    """Save current metrics to file"""
    metrics_path = Path("reports/baseline_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def compute_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict:
    """Compute F1 scores per class and macro F1"""
    f1_scores = {}

    for cls in range(num_classes):
        tp = np.sum((predictions == cls) & (labels == cls))
        fp = np.sum((predictions == cls) & (labels != cls))
        fn = np.sum((predictions != cls) & (labels == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores[f"class_{cls}"] = f1

    macro_f1 = np.mean(list(f1_scores.values()))
    return {"per_class": f1_scores, "macro": macro_f1}


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute accuracy"""
    return np.mean(predictions == labels)


class TestRegressionAccuracy:
    """Test for accuracy regression"""

    @pytest.mark.skip(reason="Requires trained model")
    def test_emotion_accuracy_no_regression(self):
        """Test emotion classification accuracy hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_acc = baseline["emotion"]["accuracy"]

        # Mock: In real implementation, run inference on validation set
        # For now, simulate current accuracy
        current_acc = 0.86  # Simulated

        # Check for regression
        degradation = baseline_acc - current_acc
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Emotion accuracy regressed by {degradation:.2%} (baseline: {baseline_acc:.2%}, current: {current_acc:.2%})"

    @pytest.mark.skip(reason="Requires trained model")
    def test_sentiment_accuracy_no_regression(self):
        """Test sentiment classification accuracy hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_acc = baseline["sentiment"]["accuracy"]

        # Mock: Simulate current accuracy
        current_acc = 0.91  # Simulated

        # Check for regression
        degradation = baseline_acc - current_acc
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Sentiment accuracy regressed by {degradation:.2%}"

    @pytest.mark.skip(reason="Requires trained model")
    def test_emotion_f1_no_regression(self):
        """Test emotion F1 score hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_f1 = baseline["emotion"]["f1_macro"]

        # Mock: Simulate current F1
        current_f1 = 0.84  # Simulated

        degradation = baseline_f1 - current_f1
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Emotion F1 regressed by {degradation:.2%}"

    @pytest.mark.skip(reason="Requires trained model")
    def test_sentiment_f1_no_regression(self):
        """Test sentiment F1 score hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_f1 = baseline["sentiment"]["f1_macro"]

        # Mock: Simulate current F1
        current_f1 = 0.90  # Simulated

        degradation = baseline_f1 - current_f1
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Sentiment F1 regressed by {degradation:.2%}"


class TestRegressionPerClass:
    """Test for per-class accuracy regression"""

    @pytest.mark.skip(reason="Requires trained model")
    def test_joy_f1_no_regression(self):
        """Test joy class F1 hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_f1 = baseline["emotion"]["per_class_f1"]["joy"]

        # Mock: Simulate current F1
        current_f1 = 0.89  # Simulated

        degradation = baseline_f1 - current_f1
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Joy F1 regressed by {degradation:.2%}"

    @pytest.mark.skip(reason="Requires trained model")
    def test_anger_f1_no_regression(self):
        """Test anger class F1 hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_f1 = baseline["emotion"]["per_class_f1"]["anger"]

        # Mock: Simulate current F1
        current_f1 = 0.83  # Simulated

        degradation = baseline_f1 - current_f1
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Anger F1 regressed by {degradation:.2%}"

    @pytest.mark.skip(reason="Requires trained model")
    def test_positive_sentiment_f1_no_regression(self):
        """Test positive sentiment F1 hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_f1 = baseline["sentiment"]["per_class_f1"]["positive"]

        # Mock: Simulate current F1
        current_f1 = 0.92  # Simulated

        degradation = baseline_f1 - current_f1
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Positive sentiment F1 regressed by {degradation:.2%}"

    @pytest.mark.skip(reason="Requires trained model")
    def test_negative_sentiment_f1_no_regression(self):
        """Test negative sentiment F1 hasn't regressed"""
        baseline = load_baseline_metrics()
        baseline_f1 = baseline["sentiment"]["per_class_f1"]["negative"]

        # Mock: Simulate current F1
        current_f1 = 0.89  # Simulated

        degradation = baseline_f1 - current_f1
        assert degradation < DEGRADATION_THRESHOLD, \
            f"Negative sentiment F1 regressed by {degradation:.2%}"


class TestRegressionValidationSet:
    """Test on mini validation set"""

    def test_validation_set_exists(self):
        """Test validation data is available"""
        assert len(VALIDATION_DATA) > 0
        assert all("text" in item for item in VALIDATION_DATA)

    def test_validation_set_diversity(self):
        """Test validation set has diverse labels"""
        emotions = set(item["emotion"] for item in VALIDATION_DATA)
        assert len(emotions) >= 3  # At least 3 different emotions

        sentiments = set(item["sentiment"] for item in VALIDATION_DATA)
        assert len(sentiments) >= 2  # At least 2 sentiments

    @pytest.mark.skip(reason="Requires trained model")
    def test_validation_set_predictions(self):
        """Test model can make predictions on validation set"""
        # Mock: In real implementation, load model and run inference
        predictions = []

        for item in VALIDATION_DATA:
            # Mock prediction
            pred = {
                "emotion": "joy",  # Simulated
                "sentiment": "positive",  # Simulated
                "confidence": 0.85
            }
            predictions.append(pred)

        assert len(predictions) == len(VALIDATION_DATA)
        assert all("emotion" in p for p in predictions)
        assert all("sentiment" in p for p in predictions)


class TestRegressionMetricsComputation:
    """Test metrics computation utilities"""

    def test_compute_accuracy_basic(self):
        """Test accuracy computation"""
        preds = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        acc = compute_accuracy(preds, labels)
        assert acc == 1.0

    def test_compute_accuracy_partial(self):
        """Test accuracy with some errors"""
        preds = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 1, 0, 1])  # One error
        acc = compute_accuracy(preds, labels)
        assert acc == 0.8

    def test_compute_f1_perfect(self):
        """Test F1 computation with perfect predictions"""
        preds = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        f1 = compute_f1(preds, labels, num_classes=3)

        assert f1["macro"] == 1.0
        assert all(score == 1.0 for score in f1["per_class"].values())

    def test_compute_f1_with_errors(self):
        """Test F1 computation with some errors"""
        preds = np.array([0, 1, 2, 0, 0, 2])  # Some errors
        labels = np.array([0, 1, 2, 0, 1, 2])
        f1 = compute_f1(preds, labels, num_classes=3)

        assert 0.0 <= f1["macro"] <= 1.0
        assert len(f1["per_class"]) == 3


class TestRegressionReporting:
    """Test regression reporting"""

    def test_save_and_load_metrics(self):
        """Test saving and loading metrics"""
        test_metrics = {
            "emotion": {"accuracy": 0.85, "f1_macro": 0.83},
            "sentiment": {"accuracy": 0.90, "f1_macro": 0.89}
        }

        # Save
        save_current_metrics(test_metrics)

        # Load
        loaded = load_baseline_metrics()

        assert loaded["emotion"]["accuracy"] == test_metrics["emotion"]["accuracy"]
        assert loaded["sentiment"]["accuracy"] == test_metrics["sentiment"]["accuracy"]

    def test_degradation_threshold_calculation(self):
        """Test degradation threshold calculation"""
        baseline = 0.85
        current = 0.83
        degradation = baseline - current

        assert degradation == 0.02
        assert degradation >= DEGRADATION_THRESHOLD  # Should trigger alert


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
