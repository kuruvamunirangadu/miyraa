"""Simple inspector for `data/processed/bootstrap`.

Prints dataset size and first few examples. Falls back gracefully if the
`datasets` library is not available.

Usage:
  python scripts/inspect_bootstrap.py --path data/processed/bootstrap --n 3
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure repo root on sys.path so this script can import local packages if needed
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/processed/bootstrap")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"Path does not exist: {p}")
        return

    if load_from_disk is None:
        print("The 'datasets' library is not installed.")
        print("Install it with: pip install datasets")
        return

    try:
        ds = load_from_disk(str(p))
    except Exception as e:
        print(f"Failed to load dataset from '{p}': {e}")
        return

    if "train" in ds:
        subset = ds["train"]
    else:
        # may be a single split
        # datasets returns a DatasetDict or Dataset; handle both
        try:
            subset = ds
        except Exception:
            print("Unexpected dataset structure")
            return

    print(f"Loaded dataset at {p}")
    try:
        length = len(subset)
    except Exception:
        length = None
    print("Size:", length)
    print(f"Printing up to {args.n} examples:\n")
    for i in range(min(args.n, length or args.n)):
        try:
            rec = subset[i]
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Failed to print record {i}: {e}")


if __name__ == "__main__":
    main()
