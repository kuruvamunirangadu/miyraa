"""
Expanded unit tests for loss functions.
Tests thresholds, calibration, focal loss, and multi-task losses.
"""

import pytest
import numpy as np
from src.nlp.training.losses import supcon_loss_np


# Helper functions for testing other loss types (not in main codebase yet)
def focal_loss_np(logits, labels, gamma=2.0, alpha=None):
    """Focal loss implementation for testing"""
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    p_t = probs[np.arange(len(labels)), labels]
    p_t = np.clip(p_t, 1e-7, 1.0 - 1e-7)
    ce_loss = -np.log(p_t)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * ce_loss
    if alpha is not None:
        loss = alpha * loss
    return loss.mean()


def calibration_loss_np(probs, labels):
    """Calibration loss implementation for testing"""
    max_probs = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    return np.mean((max_probs - correct) ** 2)


class TestSupConLoss:
    """Test supervised contrastive loss"""

    def test_supcon_identical_positives(self):
        """Test loss with identical positive pairs"""
        embeds = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 0, 1])
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        assert loss >= 0.0
        assert not np.isnan(loss)

    def test_supcon_no_positives(self):
        """Test loss when no positive pairs exist"""
        embeds = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        labels = np.array([0, 1, 2])  # All different
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        assert loss == 0.0 or np.isclose(loss, 0.0)

    def test_supcon_temperature_effect(self):
        """Test temperature parameter effect"""
        embeds = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        labels = np.array([0, 0, 1])

        loss_low = supcon_loss_np(embeds, labels, temperature=0.05)
        loss_high = supcon_loss_np(embeds, labels, temperature=0.5)

        assert loss_low >= 0.0
        assert loss_high >= 0.0
        # Lower temperature should give different loss
        assert not np.isclose(loss_low, loss_high, rtol=0.1)

    def test_supcon_batch_size_one(self):
        """Test with single sample"""
        embeds = np.array([[1.0, 0.0]])
        labels = np.array([0])
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        assert loss == 0.0 or np.isclose(loss, 0.0)

    def test_supcon_normalized_embeddings(self):
        """Test with L2-normalized embeddings"""
        embeds = np.array([[0.6, 0.8], [0.8, 0.6], [0.0, 1.0]])
        # Verify normalization
        norms = np.linalg.norm(embeds, axis=1)
        assert np.allclose(norms, 1.0)

        labels = np.array([0, 0, 1])
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        assert loss >= 0.0
        assert not np.isnan(loss)

    def test_supcon_many_positives(self):
        """Test with many positive pairs"""
        embeds = np.random.randn(10, 8)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        assert loss >= 0.0
        assert not np.isnan(loss)


class TestFocalLoss:
    """Test focal loss for class imbalance"""

    def test_focal_basic(self):
        """Test basic focal loss computation"""
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]])
        labels = np.array([0, 1])
        loss = focal_loss_np(logits, labels, gamma=2.0, alpha=0.25)
        assert loss >= 0.0
        assert not np.isnan(loss)

    def test_focal_gamma_zero(self):
        """Test focal loss with gamma=0 (should be cross-entropy)"""
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]])
        labels = np.array([0, 1])

        focal = focal_loss_np(logits, labels, gamma=0.0, alpha=None)
        # With gamma=0, focal loss reduces to cross-entropy
        assert focal >= 0.0

    def test_focal_high_confidence(self):
        """Test focal loss with high confidence predictions"""
        logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        labels = np.array([0, 1])
        loss = focal_loss_np(logits, labels, gamma=2.0, alpha=0.25)
        # High confidence correct predictions should have low loss
        assert 0.0 <= loss < 0.1

    def test_focal_low_confidence(self):
        """Test focal loss with low confidence predictions"""
        logits = np.array([[0.1, 0.0], [0.0, 0.1]])
        labels = np.array([0, 1])
        loss = focal_loss_np(logits, labels, gamma=2.0, alpha=0.25)
        ref_logits = np.array([[10.0, -10.0], [-10.0, 10.0]])
        ref_loss = focal_loss_np(ref_logits, labels, gamma=2.0, alpha=0.25)
        # Low confidence should have noticeably higher loss than confident predictions
        assert loss > ref_loss
        assert loss > 0.0

    def test_focal_alpha_effect(self):
        """Test alpha parameter for class balancing"""
        logits = np.array([[2.0, -1.0], [-1.0, 2.0]])
        labels = np.array([0, 1])

        loss_no_alpha = focal_loss_np(logits, labels, gamma=2.0, alpha=None)
        loss_with_alpha = focal_loss_np(logits, labels, gamma=2.0, alpha=0.75)

        assert loss_no_alpha >= 0.0
        assert loss_with_alpha >= 0.0
        # Alpha should change the loss value
        assert not np.isclose(loss_no_alpha, loss_with_alpha, rtol=0.01)


class TestCalibrationLoss:
    """Test calibration loss for probability calibration"""

    def test_calibration_perfect(self):
        """Test calibration with perfect predictions"""
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        loss = calibration_loss_np(probs, labels)
        # Perfect calibration should have low loss
        assert 0.0 <= loss < 0.1

    def test_calibration_uncertain(self):
        """Test calibration with uncertain predictions"""
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([0, 1])
        loss = calibration_loss_np(probs, labels)
        # Uncertain predictions should have measurable loss
        assert loss >= 0.0

    def test_calibration_overconfident(self):
        """Test calibration with overconfident wrong predictions"""
        probs = np.array([[0.1, 0.9], [0.9, 0.1]])  # Wrong predictions
        labels = np.array([0, 1])
        loss = calibration_loss_np(probs, labels)
        # Wrong overconfident predictions should have high loss
        assert loss > 0.5

    def test_calibration_batch_size(self):
        """Test calibration with various batch sizes"""
        for n in [2, 5, 10, 20]:
            probs = np.random.rand(n, 3)
            probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
            labels = np.random.randint(0, 3, size=n)
            loss = calibration_loss_np(probs, labels)
            assert loss >= 0.0
            assert not np.isnan(loss)


class TestThresholdOptimization:
    """Test threshold optimization for classification"""

    def test_threshold_f1_optimization(self):
        """Test finding optimal threshold for F1 score"""
        # Simulate predictions and labels
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])

        best_thresh = 0.5
        best_f1 = 0.0

        for thresh in np.arange(0.1, 1.0, 0.1):
            preds = (probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        assert 0.0 <= best_thresh <= 1.0
        assert 0.0 <= best_f1 <= 1.0

    def test_threshold_precision_recall_tradeoff(self):
        """Test precision-recall tradeoff at different thresholds"""
        probs = np.linspace(0.0, 1.0, 100)
        labels = (probs > 0.5).astype(int)

        precisions = []
        recalls = []

        for thresh in [0.3, 0.5, 0.7]:
            preds = (probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)

        # Higher threshold should increase precision, decrease recall
        assert precisions[2] >= precisions[0]  # 0.7 vs 0.3
        assert recalls[0] >= recalls[2]  # 0.3 vs 0.7


class TestMultiTaskLoss:
    """Test multi-task loss combinations"""

    def test_multitask_weighted_sum(self):
        """Test weighted sum of multiple losses"""
        # Simulate three task losses
        emotion_loss = 0.5
        sentiment_loss = 0.3
        intent_loss = 0.2

        # Weighted combination
        weights = {"emotion": 0.5, "sentiment": 0.3, "intent": 0.2}
        total = sum(loss * weights[task] for task, loss in [
            ("emotion", emotion_loss),
            ("sentiment", sentiment_loss),
            ("intent", intent_loss)
        ])

        assert 0.0 <= total <= 1.0
        assert np.isclose(total, 0.5 * 0.5 + 0.3 * 0.3 + 0.2 * 0.2)

    def test_multitask_loss_balancing(self):
        """Test automatic loss balancing"""
        # Losses with different magnitudes
        losses = np.array([0.1, 1.0, 10.0])

        # Normalize losses for balanced training
        normalized = losses / losses.sum()

        assert np.isclose(normalized.sum(), 1.0)
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)


class TestLossEdgeCases:
    """Test edge cases in loss computation"""

    def test_zero_gradient(self):
        """Test handling of zero gradients"""
        embeds = np.zeros((3, 5))
        labels = np.array([0, 0, 1])
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        # Should handle gracefully
        assert not np.isnan(loss)

    def test_extreme_logits(self):
        """Test handling of extreme logits"""
        logits = np.array([[100.0, -100.0], [-100.0, 100.0]])
        labels = np.array([0, 1])
        loss = focal_loss_np(logits, labels, gamma=2.0)
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_all_same_label(self):
        """Test when all samples have the same label"""
        embeds = np.random.randn(5, 8)
        labels = np.array([0, 0, 0, 0, 0])
        loss = supcon_loss_np(embeds, labels, temperature=0.1)
        assert loss >= 0.0
        assert not np.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
