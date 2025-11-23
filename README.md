# Miyraa NLP Emotion Engine

Production-ready multi-task emotion classification system with comprehensive preprocessing, augmentation, and data quality tools. Supports 11 core emotions, VAD dimensions, safety detection, style analysis, and intent recognition.

## Features

- 🎭 **11 Core Emotions**: joy, love, surprise, sadness, anger, fear, disgust, calm, excitement, confusion, neutral
- 📊 **VAD Dimensions**: Valence-Arousal-Dominance regression for each emotion
- 🛡️ **Safety Detection**: 4 categories (toxic, profane, threatening, harassment)
- ✍️ **Style Analysis**: 5 writing styles (formal, casual, assertive, empathetic, humorous)
- 🎯 **Intent Recognition**: 6 intent types (statement, question, request, command, expression, social)
- 🔧 **Text Preprocessing**: Social media, emoji, slang, hashtag normalization
- 📈 **Data Augmentation**: 5 strategies (synonym replacement, insertion, swap, deletion, back-translation)
- 🤖 **Enhanced Architecture**: Focal loss, uncertainty weighting, flexible backbone freezing
- 🎯 **Regularization**: Dropout, layer normalization, early stopping, gradient clipping
- 🔄 **5 Backbones Supported**: MiniLM-L6/L12, XtremeDistil, DistilRoBERTa, XLM-RoBERTa
- 🐳 **Docker Support**: Production containerization with health checks
- 🔒 **PII Scrubbing**: Presidio integration for enterprise-grade privacy
- 📝 **API Documentation**: Complete REST API with examples

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Generate curated emotion samples (130 samples)
python scripts\generate_curated_samples.py

# Generate validation set (52 samples)
python scripts\generate_validation_set.py

# Prepare data from HuggingFace (1000+ samples)
python scripts\prepare_data.py --from_hf go_emotions --n 1000 --out data\processed\production
```

### 3. Train Model

#### Basic Training (Frozen Backbone)

```bash
# Quick training with frozen backbone
python scripts\train_enhanced.py ^
  --backbone sentence-transformers/all-MiniLM-L6-v2 ^
  --freeze-strategy full ^
  --epochs 5 ^
  --batch-size 32 ^
  --lr 1e-3
```

#### Production Training (Recommended)

```bash
# Balanced training with focal loss and uncertainty weighting
python scripts\train_enhanced.py ^
  --backbone sentence-transformers/all-MiniLM-L6-v2 ^
  --freeze-strategy partial ^
  --freeze-layers 4 ^
  --dropout 0.3 ^
  --use-focal-loss ^
  --focal-gamma 2.0 ^
  --use-uncertainty-weighting ^
  --epochs 10 ^
  --batch-size 16 ^
  --lr 2e-5 ^
  --weight-decay 0.01 ^
  --scheduler cosine ^
  --patience 3
```

#### Maximum Performance Training

```bash
# Full fine-tuning with advanced regularization
python scripts\train_enhanced.py ^
  --backbone sentence-transformers/all-MiniLM-L12-v2 ^
  --freeze-strategy none ^
  --dropout 0.3 ^
  --head-hidden-dims 512 256 128 ^
  --use-focal-loss ^
  --focal-gamma 2.5 ^
  --emotion-weight 2.0 ^
  --epochs 15 ^
  --batch-size 8 ^
  --lr 5e-6 ^
  --weight-decay 0.01 ^
  --max-grad-norm 1.0 ^
  --scheduler plateau ^
  --patience 5
```

#### Legacy Training (Multi-task Script)

```bash
# Original multi-task training
python scripts\train_multi_task.py ^
  --data data\processed\production ^
  --epochs 3 ^
  --batch-size 16 ^
  --output outputs\
```

### 4. Export & Quantize

```bash
# Export to ONNX
python scripts\export_onnx.py --checkpoint outputs\checkpoint.pt --output outputs\model.onnx

# Quantize for faster inference
python scripts\quantize_onnx_static.py ^
  --input outputs\model.onnx ^
  --output outputs\model.quant.onnx ^
  --calibration-file data\processed\bootstrap\calibration.jsonl
```

### 5. Run API Server

```bash
# Start FastAPI server
python src\api\main.py

# Or use Docker
docker-compose up
```

## Data Quality & Preprocessing

### Curated Datasets
- **130 curated samples**: `data/processed/curated/samples.jsonl`
  - All 11 emotions, 4 safety categories, 5 styles, 6 intents
- **52 validation samples**: `data/processed/validation/samples.jsonl`
  - Edge cases, mixed emotions, sarcasm, ambiguity

### Preprocessing Tools
```python
from src.nlp.preprocessing.text_cleaner import preprocess_text

# Clean social media text
text = "OMG @user this is sooooo cool!!! 😍 #amazing"
clean = preprocess_text(text, lowercase=True, handle_emojis='convert')
# Output: "oh my god [USER] this is soo cool !! love amazing"
```

### Text Augmentation
```python
from src.nlp.preprocessing.augmentation import augment_text

# Generate variations
augmented = augment_text("I am very happy", method="sr", num_aug=3)
# Output: ["I am very joyful", "I am very cheerful", "I am very pleased"]
```

See [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) for complete documentation.

## Project Structure

```
miyraa/
├── src/
│   ├── api/                    # FastAPI REST API
│   ├── nlp/
│   │   ├── data/
│   │   │   └── taxonomy.py     # Emotion taxonomy & label mapping
│   │   ├── inference/          # Inference engines
│   │   ├── preprocessing/
│   │   │   ├── text_cleaner.py # Noisy text preprocessing
│   │   │   ├── augmentation.py # Data augmentation
│   │   │   ├── label_mapping.py
│   │   │   └── normalize.py
│   │   ├── safety/
│   │   │   └── pii_scrub.py    # Presidio PII detection
│   │   └── training/
│   │       └── losses.py       # Multi-task loss functions
│   └── server/
│       └── app.py              # Legacy server
├── scripts/
│   ├── generate_curated_samples.py    # Create curated dataset
│   ├── generate_validation_set.py     # Create validation set
│   ├── train_multi_task.py            # Multi-task training
│   ├── export_onnx.py                 # ONNX export
│   ├── quantize_onnx_static.py        # Static quantization
│   ├── profile_inference.py           # Performance profiling
│   └── prepare_data.py                # Data preparation
├── data/
│   └── processed/
│       ├── curated/                   # 130 hand-crafted samples
│       ├── validation/                # 52 validation samples
│       ├── production/                # Training data
│       └── bootstrap/                 # Calibration data
├── docs/
│   ├── PREPROCESSING.md               # Preprocessing guidelines
│   ├── DATA_IMPROVEMENTS.md           # Data quality summary
│   └── API.md                         # API documentation
├── tests/                             # Unit & integration tests
├── outputs/                           # Model checkpoints & ONNX
├── Dockerfile                         # Production container
├── docker-compose.yml                 # Container orchestration
└── requirements.txt                   # Python dependencies
```

## Documentation

- **[PREPROCESSING.md](docs/PREPROCESSING.md)**: Text normalization, augmentation, taxonomy
- **[DATA_IMPROVEMENTS.md](docs/DATA_IMPROVEMENTS.md)**: Data quality enhancements summary
- **[API.md](docs/API.md)**: REST API endpoints with examples
- **[ONNX_QUANT_BENCH.md](docs/ONNX_QUANT_BENCH.md)**: ONNX quantization benchmarks

## Technology Stack

- **Framework**: PyTorch 2.5.1 (CPU)
- **Transformers**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **API**: FastAPI 0.100.0
- **Inference**: ONNX Runtime (quantized models)
- **Privacy**: Presidio Analyzer & Anonymizer
- **NLP**: spaCy 3.5.0+
- **Containerization**: Docker + docker-compose
- **CI/CD**: GitHub Actions

## Model Architecture

Multi-task learning with shared backbone and 5 improved task-specific heads:

```
Input Text
    ↓
[Text Preprocessing]
    ↓
[Transformer Backbone] (flexible: MiniLM-L6/L12, DistilRoBERTa, XLM-RoBERTa)
    (384-dim or 768-dim embeddings)
    ↓
┌────────────┬────────────┬────────────┬────────────┬────────────┐
│  Emotions  │    VAD     │   Style    │   Intent   │   Safety   │
│  (11 cls)  │  (3 reg)   │  (5 cls)   │  (6 cls)   │  (4 cls)   │
│            │            │            │            │            │
│ [256→128]  │ [256→128]  │ [256→128]  │ [256→128]  │ [256→128]  │
│ LayerNorm  │ LayerNorm  │ LayerNorm  │ LayerNorm  │ LayerNorm  │
│ Dropout    │ Dropout    │ Dropout    │ Dropout    │ Dropout    │
│ Residual   │ Residual   │ Residual   │ Residual   │ Residual   │
│            │            │            │            │            │
│ FocalLoss  │   MSE      │   CE       │   CE       │   CE       │
└────────────┴────────────┴────────────┴────────────┴────────────┘
    ↓
[Multi-Task Loss with Uncertainty Weighting]
```

**New Features**:
- **Improved Task Heads**: Deeper architecture (256→128) with layer normalization, dropout (0.3), and residual connections
- **Focal Loss**: Handles class imbalance for rare emotions (fear, disgust, calm)
- **Uncertainty Weighting**: Automatically learns optimal task weights during training
- **Flexible Freezing**: Full, partial, or progressive backbone unfreezing strategies
- **Enhanced Regularization**: Early stopping, gradient clipping, LR scheduling

See [docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md) for details.

## API Usage

### POST /api/v1/analyze
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It made my day!"}'
```

Response:
```json
{
  "emotion": "joy",
  "confidence": 0.92,
  "vad": {"valence": 0.85, "arousal": 0.65, "dominance": 0.70},
  "style": "casual",
  "intent": "expression",
  "safety": "safe"
}
```

See [`docs/API.md`](docs/API.md) for complete API documentation.

## Documentation

- **[docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md)**: Model architecture enhancements (focal loss, uncertainty weighting, flexible freezing)
- **[docs/DATA_QUALITY.md](docs/DATA_QUALITY.md)**: Data preprocessing, augmentation, and curation guidelines
- **[docs/API.md](docs/API.md)**: Complete REST API documentation with examples
- **[docs/ONNX_QUANT_BENCH.md](docs/ONNX_QUANT_BENCH.md)**: ONNX quantization benchmarks and performance optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use Miyraa in your research, please cite:

```bibtex
@software{miyraa2025,
  title={Miyraa: Multi-Task Emotion Classification Engine},
  author={Kuruvilla Munirangadu},
  year={2025},
  url={https://github.com/kuruvamunirangadu/miyraa}
}
```

### 3) CI Torch smoke job

We added a GitHub Actions job that installs CPU PyTorch and runs the smoke training test. It's slower than the ONNX-only smoke job but verifies that the torch-based trainer works on CI (useful to catch API regressions). If you'd prefer to keep CI lightweight, remove or disable `.github/workflows/torch_smoke.yml`.

## What I need from you to proceed automatically

- Which checkpoint(s) to export / quantize (local path or HF repo id). If multimodel, confirm desired export layout: single fused ONNX or backbone + per-head ONNXs. Default: fused single ONNX.
- Whether you want static quantization with a calibration run now (if yes, confirm a path to a calibration dataset or let me sample 200 items from bootstrap). Default: sample 200 from `data/processed/bootstrap` if present.
- Whether to run the real multi-task training now. If yes, which backbone do you prefer for the short test: `sentence-transformers/all-MiniLM-L6-v2` (faster) or `xlm-roberta-base` (multilingual, slower). Default: `all-MiniLM-L6-v2` for speed.

If you prefer, I can also expand PII masking (integrate Presidio or expand the regex list). Let me know.
