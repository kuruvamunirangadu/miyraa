"""Small helper to pre-download/cache a SentenceTransformer model.

Usage:
    python scripts/cache_embedder.py --model all-MiniLM-L6-v2 --sample 8

This will import sentence_transformers, instantiate the model (causing it to
download and cache weights & tokenizer), and optionally run a tiny encode on a
few synthetic strings to warm the model (useful to avoid stalls later).
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model id")
    parser.add_argument("--sample", type=int, default=8, help="Number of synthetic texts to encode for warming")
    parser.add_argument("--quiet", action="store_true", help="Minimize output")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("sentence-transformers is not installed or failed to import:", e, file=sys.stderr)
        print("Install it with: python -m pip install sentence-transformers", file=sys.stderr)
        raise

    model_id = args.model
    if not args.quiet:
        print(f"Loading SentenceTransformer('{model_id}') to cache model files...")

    # Create model (this will download weights/tokenizer into the HF cache)
    model = SentenceTransformer(model_id)

    if args.sample and args.sample > 0:
        sample_texts = [f"This is a warmup sentence {i}." for i in range(args.sample)]
        if not args.quiet:
            print(f"Encoding {len(sample_texts)} synthetic texts to warm the model (this runs a forward pass)")
        # Run encode with a small batch to reduce memory pressure.
        emb = model.encode(sample_texts, batch_size=8, show_progress_bar=False)
        if not args.quiet:
            print("Warmup complete. Embedding shape:", getattr(emb, 'shape', 'unknown'))

    if not args.quiet:
        print("Model cache and warmup complete.")


if __name__ == '__main__':
    main()
