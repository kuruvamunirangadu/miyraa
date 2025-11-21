"""Post-training ONNX quantization utilities (dynamic quantization) and benchmark.

Usage:
  python scripts/quantize_onnx.py --input outputs/xlm-roberta.onnx --output outputs/xlm-roberta.quant.onnx --runs 20

This script uses onnxruntime.quantization.quantize_dynamic (weights-only dynamic quant)
which is a good first step for CPU/mobile deployment. It produces a smaller/faster model
and then runs a small CPU benchmark via onnxruntime.
"""
from pathlib import Path
import argparse
import time


def quantize_dynamic(input_path: str, output_path: str, weight_type="qint8"):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    qtype = QuantType.QInt8 if weight_type == "qint8" else QuantType.QUInt8
    print(f"Running dynamic quantization: {input_path} -> {output_path} (type={weight_type})")
    quantize_dynamic(model_input=input_path, model_output=output_path, weight_type=qtype)
    print("Quantization complete")


def benchmark_onnx(onnx_path: str, seq_len: int = 192, runs: int = 20):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import numpy as np

    print("Loading ONNX model for benchmark:", onnx_path)
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
    per = total / runs if runs > 0 else float("inf")
    print(f"ONNX CPU benchmark ({runs} runs): total={total:.4f}s, per_run={per:.4f}s, throughput={1.0/per:.2f} req/s")
    return total, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input ONNX not found: {inp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(str(inp), str(outp), weight_type=args.weight_type)
    benchmark_onnx(str(outp), seq_len=args.seq_len, runs=args.runs)


if __name__ == "__main__":
    main()
