# Miyraa NLP Platform — Technical Summary

**Version:** 1.0  
**Last Updated:** December 18, 2025  
**Audience:** Investors, strategic partners, demo stakeholders

---

## Platform Snapshot
- Multi-task transformer covering **11 emotions**, **VAD regression**, **intent**, **style**, and **safety** in a single forward pass (see [README.md](../README.md)).
- Exported to ONNX with quantized variants for sub-25 ms CPU inference on Pixel-class hardware (validated via [scripts/profile_inference.py](../scripts/profile_inference.py)).
- FastAPI service with health/readiness, Prometheus metrics, and Docker deployment (see [Dockerfile](../Dockerfile)).

## Architecture Highlights
- Backbone: `sentence-transformers/all-MiniLM-L6-v2` with custom heads per task ([multi_task_model.py](../src/nlp/models/multi_task_model.py)).
- Inference: ONNX Runtime with batch + single-shot APIs, automatic tokenizer warmup, and deterministic post-processing ([src/api/main.py](../src/api/main.py)).
- Android integration: SDK mapper encapsulates presentation logic and safety gating ([docs/ANDROID_UI_MAPPING.md](ANDROID_UI_MAPPING.md)).

## Privacy & Safety
- Default PII scrubbing via Presidio with regex fallback; responses strip raw text and log hashes only ([docs/PII_AND_SAFETY_GUIDE.md](PII_AND_SAFETY_GUIDE.md)).
- Safety scorer tuned to 0.35 threshold to maximize recall on risky content with contextual explanations ([reports/safety/20251218_021523/safety_summary.json](../reports/safety/20251218_021523/safety_summary.json)).
- Compliance docs cover data handling, limitations, and interpretation guidelines ([docs/KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), [docs/INTERPRETATION_GUIDE.md](INTERPRETATION_GUIDE.md)).

## Performance & Reliability
- Quantized ONNX inference targets sub-25 ms median latency on CPU; use [scripts/profile_inference.py](../scripts/profile_inference.py) to capture per-release benchmarks recorded in the reports directory.
- Prometheus metrics, request counters, and rate limiting guard against abuse ([src/api/main.py](../src/api/main.py)).
- Integration tests and privacy regression tests ensure API stability ([tests/test_api_integration.py](../tests/test_api_integration.py), [tests/test_privacy.py](../tests/test_privacy.py)).

## Deployment Readiness
- Container images built via multi-stage Dockerfile with non-root runtime user and curl healthchecks.
- CI-ready: pytest suite, mypy config, and Makefile targets for lint, test, and export.
- Documentation set includes deployment guides, evaluation playbooks, and safety quick references ([docs/DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md), [docs/EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)).

## Roadmap Signals
- Phase 5 latency profiling and quantization validated; Phase 6 privacy & compliance safeguards complete.
- Android demo builds aligned via the integration one-pager and tone mapper.
- Next milestones: multilingual expansion (pipeline stubs in [src/nlp/inference/multilang.py](../src/nlp/inference/multilang.py)) and calibration automation.
