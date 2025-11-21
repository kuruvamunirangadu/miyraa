"""Prepare a small processed dataset for local development.

This script creates a tiny synthetic dataset matching the scaffold's expected fields
and saves it to `data/processed/bootstrap` using Hugging Face `datasets`'s `save_to_disk`.

Usage:
  python scripts/prepare_data.py --n 200 --out data/processed/bootstrap

You can also attempt to bootstrap from an HF dataset (if you have network):
  python scripts/prepare_data.py --from_hf go_emotions --n 500

This script requires the `datasets` package and network access; it will fail
if `datasets` is not installed or the HF dataset cannot be downloaded.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import sys

# Ensure repo root is on sys.path so `from src...` imports work when running this script
# (useful when the package isn't installed in the venv). This locates the parent of the
# `scripts/` directory as the repository root and inserts it at the front of sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from datasets import Dataset, DatasetDict, load_dataset
except Exception:
    Dataset = None
    DatasetDict = None
    load_dataset = None

try:
    # Prefer repository taxonomy when available (full scaffold). If the `src` package
    # isn't present (e.g., when this script is copied outside the repo), fall back
    # to a small default emotion list so the script remains runnable for bootstrapping.
    from src.nlp.data.taxonomy import EMOTIONS  # type: ignore
except Exception:
    EMOTIONS = [
        "joy",
        "love",
        "calm",
        "surprise",
        "anger",
        "sadness",
        "fear",
        "disgust",
        "shame",
        "pride",
        "nostalgia",
    ]


TEMPLATES = [
    "I feel {} today.",
    "This makes me {}.",
    "I'm so {} right now.",
    "There is a sense of {} in this moment.",
]


def make_record(text: str, emotion) -> Dict:
    # emotion may be a single label str or a list of label names (multi-label)
    if isinstance(emotion, (list, tuple)):
        labs = set([e for e in emotion if e in EMOTIONS])
    else:
        labs = {emotion} if emotion in EMOTIONS else set()
    # multi-hot labels list in same order as EMOTIONS
    labels = [1 if e in labs else 0 for e in EMOTIONS]
    # rudimentary VAD mapping: sample around prototypical values per emotion
    base = {
        "joy": (0.8, 0.6, 0.6),
        "love": (0.9, 0.4, 0.7),
        "calm": (0.6, 0.1, 0.7),
        "surprise": (0.2, 0.9, 0.4),
        "anger": (0.1, 0.8, 0.5),
        "sadness": (0.1, 0.2, 0.3),
        "fear": (0.2, 0.7, 0.2),
        "disgust": (0.1, 0.6, 0.2),
        "shame": (0.2, 0.3, 0.1),
        "pride": (0.7, 0.5, 0.8),
        "nostalgia": (0.6, 0.3, 0.5),
    }
    v, a, d = base.get(emotion, (0.5, 0.5, 0.5))
    # add small noise
    v = round(min(max(random.gauss(v, 0.05), 0.0), 1.0), 3)
    a = round(min(max(random.gauss(a, 0.05), 0.0), 1.0), 3)
    d = round(min(max(random.gauss(d, 0.05), 0.0), 1.0), 3)

    return {"text": text, "labels": labels, "vad": {"v": v, "a": a, "d": d}}


def synthetic_dataset(n: int) -> List[Dict]:
    data = []
    for i in range(n):
        emo = random.choice(EMOTIONS)
        tmpl = random.choice(TEMPLATES)
        txt = tmpl.format(emo)
        data.append(make_record(txt, emo))
    return data


def save_as_hf(data: List[Dict], out: Path):
    if Dataset is None:
        raise RuntimeError("datasets library is not available")
    ds = Dataset.from_list(data)
    dd = DatasetDict({"train": ds})
    dd.save_to_disk(str(out))


def save_as_jsonl(data: List[Dict], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/processed/bootstrap")
    parser.add_argument(
        "--from_hf",
        type=str,
        default="go_emotions",
        help="HF dataset name to download",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if load_dataset is None:
        raise RuntimeError(
            "The 'datasets' library is required to run this script."
            " Install via 'pip install datasets' and retry."
        )

    # Required: attempt to download the HF dataset and fail loudly on errors
    print(f"Attempting to load HF dataset '{args.from_hf}' (n={args.n}) ...")
    try:
        d = load_dataset(args.from_hf, split="train")
    except Exception as e:
        raise RuntimeError(
            "Failed to download or load dataset '" + str(args.from_hf) + "': " + str(e)
        )

    # Map to our simple schema: text, labels (multi-hot mapped to our taxonomy), vad
    mapped = []
    # try to import mapping helper
    try:
        from src.nlp.preprocessing.label_mapping import map_goemotions_to_target
    except Exception:
        map_goemotions_to_target = None
    # try to extract label name list from dataset features if available
    label_names = None
    try:
        # HF datasets often store label names in d.features['labels']
        feat = d.features.get("labels")
        if feat is not None and hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            label_names = feat.feature.names
        elif feat is not None and hasattr(feat, "names"):
            label_names = feat.names
    except Exception:
        label_names = None

    for i, r in enumerate(d):
        if i >= args.n:
            break
        txt = r.get("text") or r.get("sentence") or r.get("content") or r.get("comment_text") or ""
        # Choose a mapping strategy: if label info exists, try to use it; otherwise random
        emo = None
        # try common label fields
        for key in ("labels", "label", "emotion", "emotions", "label_text"):
            if key in r and r[key] is not None:
                val = r[key]
                # if it's a list of indices, pick first mapped label idx
                if isinstance(val, list) and len(val) > 0:
                    # if dataset uses label indices (e.g., GoEmotions) and we have label_names,
                    # convert indices -> label names
                    if label_names is not None and len(val) == len(label_names):
                        # val here may be multi-hot binary list: pick all indices with 1
                        names = [label_names[j] for j, flag in enumerate(val) if flag]
                        if map_goemotions_to_target is not None:
                            mapped_names = map_goemotions_to_target(names, txt)
                            if mapped_names:
                                emo = mapped_names
                                break
                        # fallback: use first matched name if any
                        if names:
                            emo = names[0]
                            break
                    else:
                        # if list contains label ids (indices), pick first id as index if valid
                        try:
                            idx = int(val[0])
                            if 0 <= idx < len(EMOTIONS):
                                emo = EMOTIONS[idx]
                        except Exception:
                            pass
                elif isinstance(val, int):
                    if 0 <= val < len(EMOTIONS):
                        emo = EMOTIONS[val]
                elif isinstance(val, str):
                    # simple string match
                    s = val.lower()
                    for e in EMOTIONS:
                        if e in s:
                            emo = e
                            break
        if emo is None:
            # fallback to random mapping
            emo = random.choice(EMOTIONS)
        # if emo is a list (multi-label) pass as-is, otherwise single label
        mapped.append(make_record(txt, emo))

    # Save mapped dataset
    save_as_hf(mapped, out)
    print("Saved HF-derived dataset to", out)


if __name__ == "__main__":
    main()
