"""Multi-task trainer: emotions (multi-label BCE), VAD regression (SmoothL1), safety (BCE).

This script is a compact training harness intended for experimentation. It uses
`transformers` / `datasets` if available and falls back to synthetic data
generators otherwise. The model backbone is `xlm-roberta-base` (from Hugging Face).

Features implemented per request:
- Multi-head output: emotions (C logits), vad (3 outputs), safety (1 logit)
- Per-head learnable temperature parameters (saved to `thresholds.yaml` after
  calibration on a validation set)
- Training loop with combined loss:
    L = L_emotions (BCEWithLogits) + alpha_vad * SmoothL1(vad_pred, vad_true) + alpha_safety * BCEWithLogits(safety)
- Evaluation: uses `scripts.eval_report.eval_report` to compute macro-F1 and CCC

This is intentionally minimal and uses CPU if GPU unavailable.
"""
from pathlib import Path
import argparse
import random
import yaml
import json
import math
import sys

# Ensure repo root on sys.path so we can import local modules like scripts.eval_report
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from transformers import AutoModel, AutoTokenizer
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

from scripts.eval_report import eval_report


class MultiHeadModel(nn.Module):
    def __init__(self, backbone_dim: int, emotions_dim: int, style_dim: int = 5, intent_dim: int = 6, safety_dim: int = 4):
        super().__init__()
        # a tiny projection head on top of backbone output
        self.proj = nn.Linear(backbone_dim, backbone_dim)
        self.emotions = nn.Linear(backbone_dim, emotions_dim)
        self.vad = nn.Linear(backbone_dim, 3)
        # additional heads requested by user
        self.style = nn.Linear(backbone_dim, style_dim)
        self.intent = nn.Linear(backbone_dim, intent_dim)
        self.safety = nn.Linear(backbone_dim, safety_dim)
        # per-head temperature (learnable scalar parameters)
        self.temp_emotions = nn.Parameter(torch.tensor(1.0))
        self.temp_vad = nn.Parameter(torch.tensor(1.0))
        self.temp_style = nn.Parameter(torch.tensor(1.0))
        self.temp_intent = nn.Parameter(torch.tensor(1.0))
        self.temp_safety = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        h = torch.relu(self.proj(x))
        return {
            "emotions": self.emotions(h),
            "vad": self.vad(h),
            "style": self.style(h),
            "intent": self.intent(h),
            "safety": self.safety(h),
        }


def make_dummy_dataset(n=128, emotions_dim=28):
    # labels: multi-label random for emotions, vad floats in [0,1], safety binary
    texts = [f"sample {i}" for i in range(n)]
    emo = (np.random.rand(n, emotions_dim) > 0.8).astype(int)
    vad = np.random.rand(n, 3).astype(float)
    safety = (np.random.rand(n, 4) > 0.95).astype(int)
    style = (np.random.rand(n, 5) > 0.9).astype(int)
    intent = (np.random.rand(n, 6) > 0.9).astype(int)
    return texts, emo, vad, style, intent, safety


def encode_texts_with_backbone(texts, tokenizer, model, device, max_len=192):
    # simple encoder: get last hidden state mean pooling
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False, return_dict=True)
        # mean pool last hidden state
        last = out.last_hidden_state
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.unsqueeze(-1)
            summed = (last * attn).sum(1)
            lens = attn.sum(1).clamp_min(1)
            pooled = summed / lens
        else:
            pooled = last.mean(1)
    return pooled.cpu()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # If the user requested dummy data, skip loading the heavy transformers backbone
    if args.use_dummy or not _HAS_TRANSFORMERS:
        tokenizer = None
        backbone = None
        backbone_dim = args.backbone_dim
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        backbone = AutoModel.from_pretrained(args.backbone)
        backbone.to(device)
        backbone.eval()
        backbone_dim = backbone.config.hidden_size

    # Defer model creation until after we know the dataset/emotion dimension
    emotions_dim = args.emotions_dim
    style_dim = args.style_dim
    intent_dim = args.intent_dim
    safety_dim = args.safety_dim
    model = None

    bce_loss = nn.BCEWithLogitsLoss()
    vad_loss = nn.SmoothL1Loss()

    # prepare data
    if args.use_dummy:
        texts, emo, vad, style, intent, safety = make_dummy_dataset(args.n, emotions_dim)
        X = None
    else:
        # Attempt to load a preprocessed HF-style dataset from disk (save_to_disk output)
        try:
            from datasets import load_from_disk

            ds = load_from_disk(args.data_path)
            # expect a train split at ds['train'] or ds if it's a Dataset
            if isinstance(ds, dict) or hasattr(ds, "get") and "train" in ds:
                dd = ds["train"] if "train" in ds else ds
            else:
                dd = ds
            texts = []
            emo_list = []
            vad_list = []
            style_list = []
            intent_list = []
            safety_list = []
            for i, rec in enumerate(dd):
                if i >= args.n:
                    break
                txt = rec.get("text") or ""
                labels = rec.get("labels") or []
                # labels is expected to be a list of 0/1 values aligned with EMOTIONS
                emo_list.append(np.array(labels, dtype=int))
                vad = rec.get("vad") or {}
                v = vad.get("v", 0.5)
                a = vad.get("a", 0.5)
                d = vad.get("d", 0.5)
                vad_list.append([v, a, d])
                # style/intent/safety may be present in your dataset; try to read them
                style_list.append(np.array(rec.get("style", [0] * style_dim), dtype=int) if rec.get("style") is not None else np.zeros((style_dim,), dtype=int))
                intent_list.append(np.array(rec.get("intent", [0] * intent_dim), dtype=int) if rec.get("intent") is not None else np.zeros((intent_dim,), dtype=int))
                # safety may be multi-label of size safety_dim or a single flag
                saf = rec.get("safety")
                if isinstance(saf, list):
                    safety_list.append(np.array(saf, dtype=int))
                else:
                    safety_list.append(np.array([int(bool(saf))] + [0] * (safety_dim - 1), dtype=int))
                texts.append(txt)
            emo = np.stack(emo_list, axis=0) if emo_list else np.zeros((0, emotions_dim), dtype=int)
            vad = np.array(vad_list, dtype=float)
            style = np.stack(style_list, axis=0) if style_list else np.zeros((0, style_dim), dtype=int)
            intent = np.stack(intent_list, axis=0) if intent_list else np.zeros((0, intent_dim), dtype=int)
            safety = np.stack(safety_list, axis=0) if safety_list else np.zeros((0, safety_dim), dtype=int)
            X = None
        except Exception:
            # fallback to dummy if loading fails
            texts, emo, vad, safety = make_dummy_dataset(args.n, emotions_dim)
            X = None

    # encode texts in batches using backbone if available
    if backbone is not None:
        all_feats = []
        batch = []
        for t in texts:
            batch.append(t)
            if len(batch) >= args.encode_batch_size:
                feats = encode_texts_with_backbone(batch, tokenizer, backbone, device, max_len=args.max_len)
                all_feats.append(feats)
                batch = []
        if batch:
            feats = encode_texts_with_backbone(batch, tokenizer, backbone, device, max_len=args.max_len)
            all_feats.append(feats)
        X = torch.cat(all_feats, dim=0)
    else:
        # simple random features
        X = torch.randn(args.n, backbone_dim)

    X = X.to(device)
    y_emo = torch.tensor(emo, dtype=torch.float32, device=device)
    y_vad = torch.tensor(vad, dtype=torch.float32, device=device)
    y_style = torch.tensor(style, dtype=torch.float32, device=device)
    y_intent = torch.tensor(intent, dtype=torch.float32, device=device)
    y_safety = torch.tensor(safety, dtype=torch.float32, device=device)

    # If the loaded dataset's emotion dimension doesn't match the model's expected
    # dimension, adjust by creating the model with the correct output size.
    if model is None:
        # prefer the label size from data if available
        try:
            emotions_dim = int(y_emo.shape[1])
        except Exception:
            emotions_dim = args.emotions_dim
        # determine dims for other heads
        try:
            style_dim = int(y_style.shape[1])
        except Exception:
            style_dim = args.style_dim
        try:
            intent_dim = int(y_intent.shape[1])
        except Exception:
            intent_dim = args.intent_dim
        try:
            safety_dim = int(y_safety.shape[1])
        except Exception:
            safety_dim = args.safety_dim

        model = MultiHeadModel(backbone_dim=backbone_dim, emotions_dim=emotions_dim, style_dim=style_dim, intent_dim=intent_dim, safety_dim=safety_dim)
        model.to(device)

    # create optimizer now that model exists
    opt = optim.Adam(model.parameters(), lr=args.lr)

    dataset = list(range(args.n))

    for epoch in range(args.epochs):
        random.shuffle(dataset)
        losses = []
        for i in range(0, args.n, args.batch_size):
            idx = dataset[i : i + args.batch_size]
            xb = X[idx]
            emo_t = y_emo[idx]
            vad_t = y_vad[idx]
            style_t = y_style[idx]
            intent_t = y_intent[idx]
            saf_t = y_safety[idx]

            preds = model(xb)
            emo_logits = preds["emotions"] / model.temp_emotions
            vad_pred = preds["vad"] / model.temp_vad
            style_logits = preds["style"] / model.temp_style
            intent_logits = preds["intent"] / model.temp_intent
            saf_logits = preds["safety"] / model.temp_safety

            l_emo = bce_loss(emo_logits, emo_t)
            l_vad = vad_loss(vad_pred, vad_t)
            l_style = bce_loss(style_logits, style_t)
            l_intent = bce_loss(intent_logits, intent_t)
            l_saf = bce_loss(saf_logits, saf_t)

            loss = l_emo + args.alpha_vad * l_vad + args.alpha_style * l_style + args.alpha_intent * l_intent + args.alpha_safety * l_saf

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        print(f"Epoch {epoch+1}/{args.epochs} mean loss: {sum(losses)/len(losses):.4f}")

    # Calibration: compute per-head temperatures and per-class thresholds using a
    # small validation split. We search a small grid for a scalar temperature
    # per head (used to rescale logits) and then find the per-class threshold
    # that maximizes F1 on the validation split.
    val_idx = list(range(int(0.8 * args.n), args.n))
    with torch.no_grad():
        xb = X[val_idx]
        emo_t = y_emo[val_idx].cpu().numpy()
        vad_t = y_vad[val_idx].cpu().numpy()
        style_t = y_style[val_idx].cpu().numpy()
        intent_t = y_intent[val_idx].cpu().numpy()
        saf_t = y_safety[val_idx].cpu().numpy()
        preds = model(xb)
        emo_logits = preds["emotions"].cpu().numpy()
        vad_pred = preds["vad"].cpu().numpy()
        style_logits = preds["style"].cpu().numpy()
        intent_logits = preds["intent"].cpu().numpy()
        saf_logits = preds["safety"].cpu().numpy()

    def f1_score_binary(y_true, y_pred):
        # y_true, y_pred are binary 0/1 arrays
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        denom = (2 * tp + fp + fn)
        return 2 * tp / denom if denom > 0 else 0.0

    def find_temp_for_bce(logits, targets):
        best_t = 1.0
        best_score = -1.0
        for t in [0.5, 0.75, 1.0, 1.5, 2.0]:
            probs = 1 / (1 + np.exp(-logits / t))
            # compute macro-F1 across columns
            if probs.ndim == 1:
                score = f1_score_binary(targets, (probs >= 0.5).astype(int))
            else:
                scores = [f1_score_binary(targets[:, j], (probs[:, j] >= 0.5).astype(int)) for j in range(probs.shape[1])]
                score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_t = t
        return best_t

    def find_thresholds_by_f1(probs, targets, steps=19):
        # For each class, sweep thresholds in [0.01,0.99] to maximize F1
        ths = []
        for j in range(probs.shape[1]):
            best_th = 0.5
            best_f1 = -1.0
            pj = probs[:, j]
            yj = targets[:, j]
            for th in np.linspace(0.01, 0.99, steps):
                pbin = (pj >= th).astype(int)
                f1 = f1_score_binary(yj, pbin)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = float(th)
            ths.append(best_th)
        return ths

    # find temps
    temp_emotions = find_temp_for_bce(emo_logits, emo_t)
    temp_style = find_temp_for_bce(style_logits, style_t)
    temp_intent = find_temp_for_bce(intent_logits, intent_t)
    temp_safety = find_temp_for_bce(saf_logits, saf_t)
    temp_vad = 1.0

    # compute probs and per-class thresholds
    emo_probs = 1 / (1 + np.exp(-emo_logits / temp_emotions))
    style_probs = 1 / (1 + np.exp(-style_logits / temp_style))
    intent_probs = 1 / (1 + np.exp(-intent_logits / temp_intent))
    saf_probs = 1 / (1 + np.exp(-saf_logits / temp_safety))

    emo_thresholds = find_thresholds_by_f1(emo_probs, emo_t)
    style_thresholds = find_thresholds_by_f1(style_probs, style_t)
    intent_thresholds = find_thresholds_by_f1(intent_probs, intent_t)
    saf_thresholds = find_thresholds_by_f1(saf_probs, saf_t)

    thresholds = {
        "temp_emotions": float(temp_emotions),
        "temp_style": float(temp_style),
        "temp_intent": float(temp_intent),
        "temp_vad": float(temp_vad),
        "temp_safety": float(temp_safety),
        "emo_thresholds": emo_thresholds,
        "style_thresholds": style_thresholds,
        "intent_thresholds": intent_thresholds,
        "safety_thresholds": saf_thresholds,
    }
    Path(args.thresholds_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.thresholds_out).write_text(yaml.safe_dump(thresholds))
    print(f"Saved thresholds to {args.thresholds_out}")

    # Evaluate and save report
    # compute predictions on full set using learned temps and thresholds
    with torch.no_grad():
        preds = model(X)
        emo_logits = preds["emotions"].cpu().numpy() / thresholds["temp_emotions"]
        vad_pred = (preds["vad"].cpu().numpy() / thresholds["temp_vad"]).astype(float)
        style_logits = preds["style"].cpu().numpy() / thresholds["temp_style"]
        intent_logits = preds["intent"].cpu().numpy() / thresholds["temp_intent"]
        saf_logits = preds["safety"].cpu().numpy() / thresholds["temp_safety"]

    emo_probs = 1 / (1 + np.exp(-emo_logits))
    style_probs = 1 / (1 + np.exp(-style_logits))
    intent_probs = 1 / (1 + np.exp(-intent_logits))
    saf_probs = 1 / (1 + np.exp(-saf_logits))

    # apply per-class thresholds
    emo_preds = (emo_probs >= np.array(thresholds.get("emo_thresholds", [0.5] * emo_probs.shape[1]))).astype(int)
    style_preds = (style_probs >= np.array(thresholds.get("style_thresholds", [0.5] * style_probs.shape[1]))).astype(int)
    intent_preds = (intent_probs >= np.array(thresholds.get("intent_thresholds", [0.5] * intent_probs.shape[1]))).astype(int)
    saf_preds = (saf_probs >= np.array(thresholds.get("safety_thresholds", [0.5] * saf_probs.shape[1]))).astype(int)

    report = eval_report(y_true_emotions=(y_emo.cpu().numpy()), y_pred_emotions=emo_preds, y_true_vad=y_vad.cpu().numpy(), y_pred_vad=vad_pred)
    # add per-head summaries
    report["style_summary"] = {"shape": list(style_preds.shape)}
    report["intent_summary"] = {"shape": list(intent_preds.shape)}
    report["safety_summary"] = {"shape": list(saf_preds.shape)}

    outp = Path(args.report_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {outp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="xlm-roberta-base")
    parser.add_argument("--backbone-dim", type=int, default=768)
    parser.add_argument("--max-len", type=int, dest="max_len", default=128,
                        help="Maximum token length for backbone encoding (cap to reduce latency)")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--emotions-dim", type=int, default=11)
    parser.add_argument("--style-dim", type=int, default=5)
    parser.add_argument("--intent-dim", type=int, default=6)
    parser.add_argument("--safety-dim", type=int, default=4)
    parser.add_argument("--alpha-style", type=float, dest="alpha_style", default=0.5)
    parser.add_argument("--alpha-intent", type=float, dest="alpha_intent", default=0.5)
    parser.add_argument("--alpha-vad", type=float, dest="alpha_vad", default=1.0)
    parser.add_argument("--alpha-safety", type=float, dest="alpha_safety", default=1.0)
    parser.add_argument("--encode-batch-size", type=int, dest="encode_batch_size", default=8,
                        help="Batch size for backbone encoding to limit memory/CPU usage")
    parser.add_argument("--use-dummy", action="store_true", dest="use_dummy", default=False,
                        help="Use synthetic dummy data instead of reading a dataset")
    parser.add_argument("--thresholds-out", type=str, default="outputs/thresholds.yaml")
    parser.add_argument("--report-out", type=str, default="reports/eval_report.json")
    parser.add_argument("--data-path", type=str, dest="data_path", default="data/processed/bootstrap",
                        help="Path to HF-style dataset saved with save_to_disk")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
