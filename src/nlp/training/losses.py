"""Lightweight loss utilities.

This file implements a small supervised contrastive-style loss using numpy so
unit tests can run without a heavy torch dependency. If you prefer a PyTorch
implementation, we can add it behind a `try: import torch` guard.
"""
from typing import Sequence
import math
from typing import Optional


def supcon_loss_np(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float = 0.07,
) -> float:
    """Compute a simple supervised contrastive-like loss using cosine similarities.

    embeddings: list of vectors (list of floats)
    labels: list of integer labels (one integer per example)
    Returns average loss (float).
    """
    # convert to list of lists
    embs = [list(map(float, e)) for e in embeddings]
    n = len(embs)
    if n == 0:
        return 0.0

    # normalize
    norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in embs]
    unit = [[v[i] / norms[idx] for i in range(len(v))] for idx, v in enumerate(embs)]

    def cos(u, v):
        return sum(ua * va for ua, va in zip(u, v))

    losses = []
    for i in range(n):
        pos_idxs = [j for j in range(n) if j != i and labels[j] == labels[i]]
        if not pos_idxs:
            # no positives: skip or give zero loss
            continue
        logits = [math.exp(cos(unit[i], unit[j]) / temperature) for j in range(n) if j != i]
        # numerator: sum over positives
        numerator = sum(math.exp(cos(unit[i], unit[j]) / temperature) for j in pos_idxs)
        denom = sum(logits)
        if denom <= 0:
            continue
        loss_i = -math.log(numerator / denom)
        losses.append(loss_i)

    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))


__all__ = ["supcon_loss_np"]

# Optional PyTorch-backed implementation if torch is available. We keep the
# pure-Python `supcon_loss_np` for CI and lightweight runs, and expose a
# `supcon_loss` dispatcher that will use torch if present.
try:
    import torch
    import torch.nn.functional as F

    def supcon_loss_torch(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Compute supervised contrastive-like loss using PyTorch tensors.

        embeddings: Tensor of shape (N, D)
        labels: LongTensor of shape (N,)
        Returns scalar tensor loss.
        """
        if embeddings.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        embs = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embs, embs.t()) / temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(embeddings.device)

        # remove diagonal
        diag = torch.eye(mask.shape[0], device=embeddings.device)
        mask = mask - diag

        exp_sim = torch.exp(sim_matrix) * (1.0 - diag)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # only keep positives
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -mean_log_prob_pos
        # average over examples that had positives
        valid = (mask.sum(dim=1) > 0).float()
        if valid.sum() == 0:
            # return a zero tensor that participates in autograd so callers can .backward()
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return (loss * valid).sum() / valid.sum()

    def supcon_loss(embeddings, labels, temperature: float = 0.07):
        """Dispatcher: prefer torch implementation when available.

        Accepts either numpy-like sequences (will convert to torch) or torch tensors.
        """
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        return supcon_loss_torch(embeddings, labels, temperature)

    __all__.append("supcon_loss")
except Exception:
    # torch not available; supcon_loss will fallback to numpy version
    def supcon_loss(embeddings, labels, temperature: float = 0.07):
        return supcon_loss_np(embeddings, labels, temperature)

    __all__.append("supcon_loss")
