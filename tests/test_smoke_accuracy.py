import os
import sys
import pytest

try:
    import torch
except Exception:
    torch = None

from pathlib import Path

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_smoke_train_multi_task_creates_thresholds(tmp_path):
    """Run a tiny multi-task training with synthetic data and ensure thresholds file is created."""
    # import locally to avoid heavy CLI parsing
    from scripts.train_multi_task import train, argparse

    class Args:
        backbone = "xlm-roberta-base"
        backbone_dim = 768
        max_len = 64
        n = 200
        epochs = 1
        batch_size = 32
        lr = 1e-3
        emotions_dim = 11
        style_dim = 5
        intent_dim = 6
        safety_dim = 4
        alpha_vad = 1.0
        alpha_style = 0.5
        alpha_intent = 0.5
        alpha_safety = 1.0
        encode_batch_size = 8
        use_dummy = True
        thresholds_out = str(tmp_path / "thresholds.yaml")
        report_out = str(tmp_path / "eval.json")
        data_path = "data/processed/bootstrap"

    args = Args()
    train(args)
    assert Path(args.thresholds_out).exists(), "thresholds.yaml was not created"
    assert Path(args.report_out).exists(), "eval report not created"
