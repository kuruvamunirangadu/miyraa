import pytest


def _has_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_supcon_torch_runs():
    import torch
    from src.nlp.training.losses import supcon_loss

    embs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    loss = supcon_loss(embs, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0
