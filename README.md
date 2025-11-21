# miyraa

Lightweight development repo for preparing small local emotion-classification datasets, exporting fused ONNX models, and running static quantization. Includes a minimal CI workflow with ONNX and optional Torch smoke tests. Useful for fast prototyping, calibration, and model verification on CPU-only environments.

## Quick Start

### Prepare a small HF-derived bootstrap dataset

1. Activate your virtual environment (Windows cmd.exe):

```
.venv\Scripts\activate
```

2. Install the datasets library (if not already):

```
python -m pip install --upgrade pip
python -m pip install datasets
```

3. Run the data prep script (example with 200 examples):

```
python scripts\prepare_data.py --from_hf go_emotions --n 200 --out data\processed\bootstrap
```

4. Inspect the saved dataset:

```
python scripts\inspect_bootstrap.py --path data\processed\bootstrap --n 3
```

If you want CI to run tests and linting, push to GitHub; the included workflow will run pytest and flake8.

## New tools and workflows

We added static ONNX quantization tooling and a CI job that can run a Torch-based smoke test (CPU-only).

Key files added:

- `scripts/quantize_onnx_static.py` — performs ONNX static quantization using onnxruntime.quantization.quantize_static; accepts a small calibration file (JSONL with `text` entries) or samples from `data/processed/bootstrap`.
- `.github/workflows/torch_smoke.yml` — GitHub Actions job that installs a CPU wheel of PyTorch and runs the smoke training test (`tests/test_smoke_accuracy.py`). This is slower than the ONNX smoke check, so it's optional; keep it if you want full torch verification in CI.
- `requirements.txt` — a minimal set of Python dependencies (note: for CPU PyTorch we install in CI using the official CPU wheel index; do not add a GPU wheel to CI unless you need GPUs).

## Quick usage

### 1) Static quantization (uses calibration samples):

Prepare a calibration file: a JSONL file with one JSON object per line containing a `text` field. Example:

	{"text": "I love this product"}
	{"text": "This is terrible"}

Then run:

```
python scripts\quantize_onnx_static.py --input outputs\my_model.onnx --output outputs\my_model.quant.static.onnx --calibration-file data\processed\bootstrap\calibration.jsonl --samples 200
```

If you don't provide a calibration file, the script will try to sample `--samples` items from `data/processed/bootstrap` (if present).

### 2) Export layout choice (single fused vs backbone+heads):

Option A (recommended for inference): export a single fused ONNX containing the backbone and all heads. This is easiest for inference because you only need to load one artifact. Use `--fused` on the export script if available.

Option B (modular): export backbone and heads separately (backbone ONNX + small per-head ONNXs). This is useful if you want to swap heads or conserve memory on-device by loading only required heads.

Defaults used by the new static quantization script assume Option A (fused) unless you pass separate files.

### 3) CI Torch smoke job

We added a GitHub Actions job that installs CPU PyTorch and runs the smoke training test. It's slower than the ONNX-only smoke job but verifies that the torch-based trainer works on CI (useful to catch API regressions). If you'd prefer to keep CI lightweight, remove or disable `.github/workflows/torch_smoke.yml`.

## What I need from you to proceed automatically

- Which checkpoint(s) to export / quantize (local path or HF repo id). If multimodel, confirm desired export layout: single fused ONNX or backbone + per-head ONNXs. Default: fused single ONNX.
- Whether you want static quantization with a calibration run now (if yes, confirm a path to a calibration dataset or let me sample 200 items from bootstrap). Default: sample 200 from `data/processed/bootstrap` if present.
- Whether to run the real multi-task training now. If yes, which backbone do you prefer for the short test: `sentence-transformers/all-MiniLM-L6-v2` (faster) or `xlm-roberta-base` (multilingual, slower). Default: `all-MiniLM-L6-v2` for speed.

If you prefer, I can also expand PII masking (integrate Presidio or expand the regex list). Let me know.
