from src.nlp.training.losses import supcon_loss_np


def test_supcon_simple():
    # two pairs of identical embeddings with same labels -> low loss
    embs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    labels = [0, 0, 1, 1]
    loss = supcon_loss_np(embs, labels)
    assert loss >= 0.0
    # since positives are identical, expect small loss (close to 0)
    assert loss < 1.0


def test_supcon_no_positives():
    embs = [[1.0, 0.0], [0.0, 1.0]]
    labels = [0, 1]
    loss = supcon_loss_np(embs, labels)
    assert loss == 0.0
