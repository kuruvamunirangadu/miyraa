"""Tiny example training loop using PyTorch and the supcon loss.

This script uses the synthetic dataset generator in `scripts.prepare_data.synthetic_dataset`
to create a tiny dataset, builds a tiny encoder (a single linear projection), and runs
a few SGD steps using the torch-backed `supcon_loss` if available.

Run locally (requires torch):
  python scripts\train_tiny.py --epochs 5 --batch-size 8
"""
from pathlib import Path
import argparse
import random
import sys
import os
import numpy as np

# Ensure repo root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    raise RuntimeError("This training script requires PyTorch. Install torch and retry.")

from src.nlp.training.losses import supcon_loss
from scripts.prepare_data import synthetic_dataset

# Delay importing sentence-transformers until it's actually requested to avoid
# expensive imports / metadata issues at module import time.
SentenceTransformer = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class TinyEncoder(nn.Module):
    def __init__(self, input_dim: int = 16, embed_dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


def make_synthetic_features(n, input_dim=16, use_embedder: str = None, embed_batch_size: int = 8):
    # Create deterministic synthetic float vectors from text content
    data = synthetic_dataset(n)
    texts = [rec["text"] for rec in data]
    labels = []
    for rec in data:
        try:
            lab = rec["labels"].index(1)
        except Exception:
            lab = 0
        labels.append(lab)

    if use_embedder:
        # allow short names like `all-MiniLM-L6-v2` and expand them to the
        # sentence-transformers namespace so callers don't need to pass the
        # fully-qualified HF id every time.
        if "/" not in use_embedder:
            use_embedder = f"sentence-transformers/{use_embedder}"
        # Prefer using HuggingFace transformers (AutoModel) for embedding to avoid
        # importing the `sentence_transformers` package which can pull heavy deps.
        try:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained(use_embedder)
            m = AutoModel.from_pretrained(use_embedder)
            m.eval()
            embs = []
            import torch as _torch
            for i in range(0, len(texts), embed_batch_size):
                batch_texts = texts[i : i + embed_batch_size]
                inputs = tok(batch_texts, return_tensors="pt", padding=True, truncation=True)
                with _torch.no_grad():
                    out = m(**inputs, return_dict=True)
                    last = out.last_hidden_state
                    attn = inputs.get("attention_mask")
                    if attn is not None:
                        attn = attn.unsqueeze(-1)
                        summed = (last * attn).sum(1)
                        lens = attn.sum(1).clamp_min(1)
                        pooled = (summed / lens).cpu().numpy()
                    else:
                        pooled = last.mean(1).cpu().numpy()
                    embs.append(pooled)
            embs = np.vstack(embs)
            return torch.tensor(embs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
        except Exception as e:
            print(f"Failed to use transformers-based embedder '{use_embedder}': {e}")
            print("Attempting to import sentence-transformers as fallback...")
            try:
                from sentence_transformers import SentenceTransformer as _ST

                model = _ST(use_embedder)
                embs = model.encode(texts, batch_size=embed_batch_size, convert_to_numpy=True)
                return torch.tensor(embs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
            except Exception as e2:
                print(f"Failed to use sentence-transformers embedder '{use_embedder}': {e2}")
                print("Falling back to deterministic synthetic features.")

    # fallback: basic deterministic vectors
    X = []
    for i, t in enumerate(texts):
        h = sum(ord(c) for c in t) % 100
        vec = [(h + j) % 7 / 7.0 for j in range(input_dim)]
        X.append(vec)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def train(args):
    torch.manual_seed(0)
    use_embedder = args.embedder

    # prepare data
    X, y = make_synthetic_features(args.n, input_dim=args.input_dim, use_embedder=use_embedder)
    input_dim = X.shape[1]

    model = TinyEncoder(input_dim=input_dim, embed_dim=args.embed_dim)

    if args.optimizer.lower() == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = optim.SGD(model.parameters(), lr=args.lr)

    dataset = list(zip(X, y))
    losses_over_time = []

    for epoch in range(args.epochs):
        random.shuffle(dataset)
        losses = []
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i : i + args.batch_size]
            xb = torch.stack([b[0] for b in batch])
            yb = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)

            emb = model(xb)
            loss = supcon_loss(emb, yb, temperature=args.temperature)
            if isinstance(loss, torch.Tensor):
                lval = loss.item()
            else:
                lval = float(loss)

            opt.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward()
                opt.step()
            else:
                opt.step()
            losses.append(lval)

        mean_loss = sum(losses) / len(losses) if losses else 0.0
        losses_over_time.append(mean_loss)
        print(f"Epoch {epoch+1}/{args.epochs} mean loss: {mean_loss:.4f}")

    # Save checkpoint if requested
    if args.checkpoint:
        outp = Path(args.checkpoint)
        outp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, str(outp))
        print(f"Saved checkpoint to {outp}")

    # Plot if requested
    if args.plot and plt is not None:
        # If the user passed a path with directories, respect it. Otherwise save under reports/
        p = Path(args.plot)
        if p.parent == Path('.') or str(p.parent) == '':
            report_dir = Path("reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            pth = report_dir / p.name
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            pth = p
        plt.figure()
        plt.plot(range(1, len(losses_over_time) + 1), losses_over_time, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.title("Training Loss")
        plt.savefig(str(pth))
        print(f"Saved loss plot to {pth}")
    elif args.plot and plt is None:
        print("matplotlib not available; skipping plot save")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"]) 
    parser.add_argument("--input-dim", type=int, dest="input_dim", default=16)
    parser.add_argument("--embed-dim", type=int, dest="embed_dim", default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Batch size for embedding encode to limit memory/CPU usage")
    parser.add_argument("--checkpoint", type=str, default="outputs/tiny_checkpoint.pt", help="Path to save model checkpoint (optional)")
    parser.add_argument("--plot", type=str, default="loss.png", help="Filename under reports/ to save loss plot (optional)")
    parser.add_argument("--embedder", type=str, default="", help="Optional sentence-transformers model id (e.g. all-MiniLM-L6-v2)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
