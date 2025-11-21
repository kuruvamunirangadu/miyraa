"""Evaluation helpers: macro-F1 for multi-label emotions and CCC for VAD regression.

This module keeps dependencies light: it will use scikit-learn if available for f1
calculations, otherwise falls back to a simple implementation. CCC is implemented
directly.

Usage:
  from scripts.eval_report import macro_f1, ccc, eval_report

Functions:
  - macro_f1(y_true, y_pred): y_true/y_pred are binary 2D arrays (num_samples, num_classes)
  - ccc(y_true, y_pred): y_true/y_pred are 2D arrays (num_samples, 3) for V,A,D
  - eval_report(...) prints and returns a summary dict
"""
from typing import Dict, Any
import numpy as np
import json
from pathlib import Path

try:
    from sklearn.metrics import f1_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 for multi-label predictions.

    y_true and y_pred must be binary matrices (N, C).
    """
    if y_true.size == 0:
        return 0.0
    if _HAS_SKLEARN:
        # average='macro' treats classes equally
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # fallback simple per-class f1
    C = y_true.shape[1]
    f1s = []
    for c in range(C):
        y_t = y_true[:, c]
        y_p = y_pred[:, c]
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance correlation coefficient across all values (flattened).

    y_true and y_pred are (N, D) arrays.
    Returns mean CCC across dimensions.
    """
    if y_true.size == 0:
        return 0.0
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    D = y_true.shape[1]
    cccs = []
    for d in range(D):
        t = y_true[:, d]
        p = y_pred[:, d]
        mean_t = t.mean()
        mean_p = p.mean()
        var_t = t.var()
        var_p = p.var()
        cov = ((t - mean_t) * (p - mean_p)).mean()
        denom = var_t + var_p + (mean_t - mean_p) ** 2
        if denom == 0:
            c = 0.0
        else:
            c = 2 * cov / denom
        cccs.append(c)
    return float(np.mean(cccs))


def eval_report(emotions_true, emotions_pred, vad_true, vad_pred, outpath: str = None) -> Dict[str, Any]:
    """Compute and optionally save evaluation report.

    - emotions_true/pred: binary arrays (N, C)
    - vad_true/pred: float arrays (N, 3)
    """
    emo_f1 = macro_f1(np.asarray(emotions_true), np.asarray(emotions_pred))
    vad_ccc = ccc(np.asarray(vad_true), np.asarray(vad_pred))
    report = {
        "emotions_macro_f1": float(emo_f1),
        "vad_ccc": float(vad_ccc),
    }
    if outpath:
        p = Path(outpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    # tiny smoke test
    y_t = np.array([[1, 0, 1], [0, 1, 0]])
    y_p = np.array([[1, 0, 0], [0, 1, 1]])
    v_t = np.array([[0.2, 0.3, 0.4], [0.6, 0.5, 0.6]])
    v_p = v_t + 0.01
    print(eval_report(y_t, y_p, v_t, v_p))
