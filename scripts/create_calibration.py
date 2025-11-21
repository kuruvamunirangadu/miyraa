#!/usr/bin/env python3
"""Create a calibration JSONL from `data/processed/bootstrap` (HF dataset on-disk).

Writes `data/processed/bootstrap/calibration.jsonl` with up to --samples lines containing {"text": ...}
"""
import argparse
from pathlib import Path
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", default="data/processed/bootstrap")
    p.add_argument("--out", default="data/processed/bootstrap/calibration.jsonl")
    p.add_argument("--samples", type=int, default=200)
    args = p.parse_args()

    from datasets import load_from_disk

    path = Path(args.bootstrap)
    if not path.exists():
        print(f"bootstrap path not found: {path}")
        return 2

    ds = load_from_disk(str(path))

    # choose dataset split
    if hasattr(ds, "keys"):
        if "train" in ds:
            d = ds["train"]
        else:
            d = list(ds.values())[0]
    else:
        d = ds

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with outp.open("w", encoding="utf-8") as fh:
        for ex in d:
            if isinstance(ex, dict) and "text" in ex:
                fh.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")
                n += 1
                if n >= args.samples:
                    break

    print(f"Wrote {n} samples to {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
