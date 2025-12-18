# Miyraa NLP Architecture Overview

**Version:** 1.0  
**Last Updated:** December 18, 2025

The diagram below summarizes the production data flow from raw text to Android UI delivery.

```mermaid
graph TD
    subgraph Data
        A[Raw Conversation Data]
        B[Preprocessing Pipelines\n(text_cleaner.py, augmentation.py)]
        C[Curated & Production Datasets\n(data/processed)]
    end

    subgraph Training
        D[Multi-task Trainer\n(train_enhanced.py / train_multi_task.py)]
        E[Best Checkpoints\n(outputs/*.pt)]
    end

    subgraph Export & Validation
        F[ONNX Export\n(export_onnx.py)]
        G[Quantization\n(quantize_onnx*.py)]
        H[Latency Profiling\n(profile_inference.py)]
        I[Safety Calibration\n(reports/safety/...)]
    end

    subgraph Serving
        J[FastAPI Service\n(src/api/main.py)]
        K[PII Scrubbing + Safety Scoring\n(src/nlp/safety/*)]
        L[Prometheus Metrics\n(/metrics)]
        M[Docker Runtime\n(Dockerfile)]
    end

    subgraph Clients
        N[Miyraa Android SDK\n(sdk/android)]
        O[NlpUiMapper\n(NlpUiMapping.kt)]
        P[KlynAI Voice + UI]
    end

    A --> B --> C --> D --> E --> F --> G --> J
    G --> H --> J
    E --> I --> J
    J --> K
    J --> L
    J --> M
    J --> N --> O --> P
```

**Key References:**
- Model definition and heads: [src/nlp/models/multi_task_model.py](../../src/nlp/models/multi_task_model.py)
- Safety + privacy pipeline: [docs/PII_AND_SAFETY_GUIDE.md](../PII_AND_SAFETY_GUIDE.md)
- Deployment workflow: [docs/DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
