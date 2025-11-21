#!/usr/bin/env python3
"""
Static ONNX quantization helper using onnxruntime.quantization.quantize_static.

Usage examples:

  python scripts\quantize_onnx_static.py --input outputs/my_model.onnx --output outputs/my_model.static_quant.onnx --calibration-file data/processed/bootstrap/calibration.jsonl --samples 200

If --calibration-file is not provided the script will attempt to sample `--samples` items from data/processed/bootstrap.
Calibration file format: JSONL with one JSON object per line containing a "text" field.
"""
import argparse
import json
import os
import random
import sys
from typing import List, Dict

try:
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
except Exception as e:
    print("onnxruntime.quantization is required. Please install onnxruntime and onnxruntime-tools.")
    raise

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class SimpleDataReader(CalibrationDataReader):
    def __init__(self, inputs: List[Dict[str, any]], input_name: str):
        self.inputs = inputs
        self.input_name = input_name
        self.iterator = iter(self.inputs)

    def get_next(self):
        try:
            batch = next(self.iterator)
            return {self.input_name: batch}
        except StopIteration:
            return None


def load_calibration_texts(calibration_file: str, samples: int) -> List[str]:
    texts = []
    if calibration_file and os.path.exists(calibration_file):
        with open(calibration_file, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        texts.append(obj["text"])
                except Exception:
                    continue
    return texts[:samples]


def sample_from_bootstrap(bootstrap_dir: str, samples: int) -> List[str]:
    # look for JSONL files under bootstrap_dir
    texts = []
    if not os.path.isdir(bootstrap_dir):
        return texts
    for fname in os.listdir(bootstrap_dir):
        if fname.endswith(".jsonl") or fname.endswith(".json"):
            path = os.path.join(bootstrap_dir, fname)
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            texts.append(obj["text"])
                    except Exception:
                        continue
    random.shuffle(texts)
    return texts[:samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model to quantize")
    parser.add_argument("--output", required=True, help="Output quantized ONNX model path")
    parser.add_argument("--calibration-file", default=None, help="JSONL file with calibration texts (one JSON per line with `text` field)")
    parser.add_argument("--samples", type=int, default=200, help="Number of calibration samples to use")
    parser.add_argument("--bootstrap-dir", default="data/processed/bootstrap", help="Fallback bootstrap dir to sample calibration texts from")
    parser.add_argument("--input-name", default=None, help="Name of the input tensor expected by the model (optional; will try to infer)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input model not found: {args.input}")
        sys.exit(2)

    texts = load_calibration_texts(args.calibration_file, args.samples)
    if len(texts) < args.samples:
        # try bootstrap
        more = sample_from_bootstrap(args.bootstrap_dir, args.samples - len(texts))
        texts += more

    if len(texts) == 0:
        print("No calibration texts available. Provide --calibration-file or ensure data/processed/bootstrap contains JSONL with `text` fields.")
        sys.exit(2)

    if AutoTokenizer is None:
        print("transformers not available; install transformers to run calibration (it is used only for tokenization).")
        sys.exit(2)

    # Choose a tokenizer by trying to detect model from the input name; fallback to a small model for tokenization
    # The safest default is a small, fast tokenizer
    tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Using tokenizer {tokenizer_name} for calibration tokenization (only for shaping inputs).")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Tokenize calibration texts into model input ids arrays
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    # Pick input_name
    input_name = args.input_name
    if input_name is None:
        # guess common names
        for candidate in ["input_ids", "input.1", "input_ids:0", "input"]:
            if candidate in tokenized:
                input_name = candidate
                break
        # fallback to the first array name
        if input_name is None:
            input_name = list(tokenized.keys())[0]

    # Build list of numpy arrays in the same shape as model expects
    inputs = []
    # calibration expects iterator of dicts mapping input_name -> numpy array inputs
    import numpy as _np

    # Create per-example inputs (onnxruntime static quant expects batches)
    for i in range(len(texts)):
        single = {k: v[i : i + 1] for k, v in tokenized.items()}
        # prefer the chosen input name
        if args.input_name and args.input_name not in single:
            # try to map first field
            single = {list(single.keys())[0]: list(single.values())[0]}
        inputs.append(single[list(single.keys())[0]])

    # Convert inputs into the structure expected by CalibrationDataReader
    # onnxruntime.quantization expects the reader to yield dict(input_name -> ndarray)
    # We'll feed batches of 1
    data_reader_inputs = []
    for i in range(len(texts)):
        # take arrays from tokenized and ensure numpy dtype
        arr = {k: v[i : i + 1] for k, v in tokenized.items()}
        # pick the first key as the input_name if not present
        if args.input_name and args.input_name in arr:
            data_reader_inputs.append(arr[args.input_name])
        else:
            data_reader_inputs.append(list(arr.values())[0])

    # Build a minimal CalibrationDataReader: we'll hand the numpy arrays to it
    # Note: quantize_static will require the correct input_name matching the model; if it fails, pass --input-name explicitly.
    class _Reader(CalibrationDataReader):
        def __init__(self, items, name):
            self.items = items
            self.name = name
            self.index = 0

        def get_next(self):
            if self.index >= len(self.items):
                return None
            v = self.items[self.index]
            self.index += 1
            return {self.name: v}

    reader = _Reader(data_reader_inputs, input_name)

    print(f"Running static quantization: input={args.input} output={args.output} samples={len(data_reader_inputs)} input_name={input_name}")

    # Call quantize_static with a conservative set of args compatible with multiple onnxruntime versions
    quantize_static(model_input=args.input,
                    model_output=args.output,
                    calibration_data_reader=reader,
                    quant_format=None,
                    per_channel=False,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8)

    print("Static quantization finished.")


if __name__ == "__main__":
    main()
