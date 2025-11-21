#!/usr/bin/env python3
"""Small ONNX benchmark helper.

Usage: python scripts/bench_onnx.py --model outputs/xlm-roberta.quant.onnx --seq-len 64 --runs 20
"""
import argparse
import time

def bench(onnx_path: str, seq_len: int = 192, runs: int = 20):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import numpy as np

    print(f"Loading ONNX model for benchmark: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    sample = "Benchmarking sentence." * 4
    enc = tok(sample, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"].astype("int64")
    attention_mask = enc["attention_mask"].astype("int64")

    inp = {sess.get_inputs()[0].name: input_ids, sess.get_inputs()[1].name: attention_mask}

    # Warmup
    for _ in range(3):
        _ = sess.run(None, inp)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, inp)
    t1 = time.perf_counter()

    total = t1 - t0
    per = total / runs if runs > 0 else float('inf')
    print(f"ONNX CPU benchmark ({runs} runs, seq_len={seq_len}): total={total:.4f}s, per_run={per:.4f}s, throughput={1.0/per:.2f} req/s")
    return total, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()
    bench(args.model, seq_len=args.seq_len, runs=args.runs)


if __name__ == '__main__':
    main()
