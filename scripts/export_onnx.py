"""Export a HF transformer model to ONNX (CPU-targeted) and run a tiny CPU benchmark.

Usage example:
  python scripts/export_onnx.py --model xlm-roberta-base --output outputs/xlm-roberta.onnx --seq-len 192

This script uses torch and onnxruntime. It creates a dummy input with seq_len tokens and
exports with dynamic axes for batch size (and optionally seq length).
"""
from pathlib import Path
import argparse
import time
import numpy as np


def export_to_onnx(model_id_or_path: str, output_path: str, seq_len: int = 192, opset: int = 12):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # model_id_or_path can be either a HF model id or a local path to a fine-tuned checkpoint
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    model.eval()

    # Prepare dummy inputs using tokenizer
    sample = "This is a warmup sentence for ONNX export."
    enc = tok(sample, return_tensors="pt", padding="max_length", truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def benchmark_onnx(onnx_path: str, model_id_or_path: str, seq_len: int = 192, runs: int = 50):
    import onnxruntime as ort
    # Avoid heavy imports during batch-mode benchmarking where possible; we still
    # use the tokenizer to build a realistic input tensor shape.
    from transformers import AutoTokenizer

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(model_id)

    sample = "Benchmarking sentence." * 5
    enc = tok(sample, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"].astype("int64")
    attention_mask = enc["attention_mask"].astype("int64")

    inp = {sess.get_inputs()[0].name: input_ids, sess.get_inputs()[1].name: attention_mask}

    # Warmup
    for _ in range(5):
        _ = sess.run(None, inp)

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, inp)
    t1 = time.perf_counter()

    total = t1 - t0
    per = total / runs
    print(f"ONNX CPU benchmark ({runs} runs): total={total:.4f}s, per_run={per:.4f}s, throughput={1.0/per:.2f} req/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="xlm-roberta-base", help="Model id or local checkpoint dir")
    p.add_argument("--output", type=str, default="outputs/model.onnx")
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--skip-benchmark", action="store_true", help="Only export ONNX and skip onnxruntime benchmark")
    args = p.parse_args()
    export_to_onnx(args.model, args.output, seq_len=args.seq_len)
    if not args.skip_benchmark:
        try:
            benchmark_onnx(args.output, args.model, seq_len=args.seq_len, runs=args.runs)
        except Exception as e:
            print(f"ONNX benchmark failed: {e}. You can run the benchmark manually with onnxruntime installed.")


if __name__ == "__main__":
    main()
