ONNX export & quantization
===========================

Quick guide to export, quantize and benchmark ONNX models for Android/CPU targets.

Export (example):

```bash
python scripts/export_onnx.py --model xlm-roberta-base --output outputs/xlm-roberta.onnx --seq-len 192
```

Dynamic quantization (weights-only):

```bash
python scripts/quantize_onnx.py --input outputs/xlm-roberta.onnx --output outputs/xlm-roberta.quant.onnx --runs 20
```

Notes:
- The `quantize_onnx.py` script performs dynamic quantization using onnxruntime's quantize_dynamic API and runs a small CPU benchmark.
- For Android deployment, consider reducing `seq_len` to 64 or 128 to lower latency and memory. Quantization reduces model size and improves CPU throughput; static quantization (with calibration) may yield better results but requires calibration datasets.
- For mobile, test quantized models with onnxruntime-mobile or NNAPI on the target device.
