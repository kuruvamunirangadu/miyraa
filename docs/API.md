# Miyraa NLP Emotion Engine - API Documentation

## Overview

The Miyraa API provides emotion classification, VAD (Valence-Arousal-Dominance) analysis representing the overall emotional state of the text, and safety detection for text inputs. Built with FastAPI, it supports both PyTorch and ONNX inference backends.

Each response returns a primary emotion label alongside the full multi-label distribution in emotion_scores, plus a single global VAD triple computed for the entire input text.


### Privacy Defaults

- `/nlp/emotion/fingerprint` scrubs PII on every request. Presidio is used when installed and the service transparently falls back to the hardened regex scrubber otherwise.
- Service logs store only hashed client identifiers and payload length metrics; raw text never appears in structured logs.
- Responses strip any `text`, `raw_text`, or `processed_text` fields before returning and include hashed PII metadata for auditability instead of sensitive content.


**Base URL:** `http://localhost:8000`

---

## Endpoints

### 1. Health Check

Check if the API is running and healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

**Example (curl):**
```bash
curl -X GET http://localhost:8000/health
```

**Example (Python):**
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

---

### 2. Analyze Emotion

Analyze text for emotions, overall VAD state, and safety classification.

**Endpoint:** `POST /api/v1/analyze`

**Request Body:**
```json
{
  "text": "I'm so excited about this amazing opportunity!",
  "scrub_pii": true,
  "return_vad": true,
  "return_safety": true
}
```

**Parameters:**
- `text` (string, required): Input text to analyze (max 512 characters recommended)
- `scrub_pii` (boolean, optional): Whether to scrub PII before analysis (default: false)
- `return_vad` (boolean, optional): Include the overall VAD triple (valence, arousal, dominance) computed once per input (default: true)
- `return_safety` (boolean, optional): Include safety classification (default: true)

**Response:**
```json
{
  "text": "I'm so excited about this amazing opportunity!",
  "emotion": "joy",
  "emotion_scores": {
    "joy": 0.95,
    "excitement": 0.88,
    "admiration": 0.72,
    "optimism": 0.65,
    "surprise": 0.45,
    "love": 0.12,
    "fear": 0.02,
    "anger": 0.01,
    "sadness": 0.01,
    "disgust": 0.01,
    "neutral": 0.05
  },
  "vad": {
    "valence": 0.89,
    "arousal": 0.76,
    "dominance": 0.54
  },
  "safety": {
    "safe": true,
    "score": 0.98,
    "categories": {
      "toxic": 0.01,
      "profane": 0.00,
      "threatening": 0.00,
      "harassment": 0.01
    }
  },
  "processing_time_ms": 45.3,
  "pii_scrubbed": false
}
```

  Dominant emotion lives in the emotion field; read emotion_scores for the full multi-label distribution.

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling great today!",
    "scrub_pii": false,
    "return_vad": true,
    "return_safety": true
  }'
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/api/v1/analyze"
payload = {
    "text": "This product is absolutely terrible!",
    "scrub_pii": False,
    "return_vad": True,
    "return_safety": True
}

response = requests.post(url, json=payload)
result = response.json()

primary = result["emotion"]
confidence = result["emotion_scores"].get(primary)

print(f"Primary emotion: {primary} (confidence {confidence:.2f})")
print(f"VAD Valence: {result['vad']['valence']:.2f}")
print(f"Is safe: {result['safety']['safe']}")
```

**Example (JavaScript/Fetch):**
```javascript
fetch('http://localhost:8000/api/v1/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'I love this new feature!',
    scrub_pii: false,
    return_vad: true,
    return_safety: true
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### Emotion Output Interpretation

- The model always produces a full multi-label distribution in `emotion_scores` (one probability per emotion).
- The `emotion` field is a convenience summary equal to the highest-probability label in `emotion_scores`.
- The confidence for the primary emotion is `emotion_scores[emotion]`; clients needing rich output should read directly from the distribution.
- VAD represents the global VAD per text triple, is computed once per input, and remains independent of the emotion label distribution.
- You can safely expose only the dominant emotion in downstream products while retaining multi-emotion data for analytics.

> **Answering the big question**: The API returns the whole probability distribution *and* a dominant label. Check `emotion_scores` when you need multiple emotions; rely on `emotion` when you only need the top result.

Emotion labels answer what is felt; VAD describes how strongly and in what direction.

---

### 3. Batch Analysis

Analyze multiple texts in a single request for better throughput.

**Endpoint:** `POST /api/v1/batch`

**Request Body:**
```json
{
  "texts": [
    "I'm so happy today!",
    "This is frustrating.",
    "Feeling calm and peaceful."
  ],
  "scrub_pii": false,
  "return_vad": true,
  "return_safety": true
}
```

**Parameters:**
- `texts` (array of strings, required): List of texts to analyze (max 100 per request)
- `scrub_pii` (boolean, optional): Whether to scrub PII (default: false)
- `return_vad` (boolean, optional): Include the overall VAD triple (valence, arousal, dominance) computed once per input (default: true)
- `return_safety` (boolean, optional): Include safety scores (default: true)

**Response:**
```json
{
  "results": [
    {
      "text": "I'm so happy today!",
      "emotion": "joy",
      "emotion_scores": {...},
      "vad": {...},
      "safety": {...}
    },
    {
      "text": "This is frustrating.",
      "emotion": "anger",
      "emotion_scores": {...},
      "vad": {...},
      "safety": {...}
    },
    {
      "text": "Feeling calm and peaceful.",
      "emotion": "calm",
      "emotion_scores": {...},
      "vad": {...},
      "safety": {...}
    }
  ],
  "total_processing_time_ms": 120.5,
  "count": 3
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Amazing work!",
      "I am disappointed.",
      "This is okay."
    ]
  }'
```

---

### 4. PII Scrubbing (Standalone)

Scrub PII from text without performing emotion analysis.

**Endpoint:** `POST /api/v1/scrub-pii`

**Request Body:**
```json
{
  "text": "Contact me at john.doe@example.com or call +1-555-123-4567",
  "use_presidio": true
}
```

**Parameters:**
- `text` (string, required): Input text to scrub
- `use_presidio` (boolean, optional): Use Presidio for enterprise-grade detection (default: true if available)

**Response:**
```json
{
  "original_length": 62,
  "scrubbed_text": "Contact me at [EMAIL] or call [PHONE]",
  "scrubbed_length": 38,
  "pii_detected": [
    {
      "type": "EMAIL_ADDRESS",
      "replaced_with": "[EMAIL]",
      "confidence": 0.95
    },
    {
      "type": "PHONE_NUMBER",
      "replaced_with": "[PHONE]",
      "confidence": 0.88
    }
  ],
  "method": "presidio"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/v1/scrub-pii \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My SSN is 123-45-6789 and email is test@example.com"
  }'
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Bad Request",
  "message": "Text field is required",
  "status_code": 400
}
```

---

## Deprecated Server

The legacy FastAPI entrypoint in server/app.py is deprecated. Use the REST endpoints documented above (served from src/api/main.py) for all new integrations.

### 422 Validation Error
```json
{
  "error": "Validation Error",
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal Server Error",
  "message": "Model inference failed",
  "status_code": 500
}
```

---

## Rate Limiting

- Default: 100 requests per minute per IP
- Batch endpoint: 20 requests per minute per IP
- Headers included in response:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

---

## Authentication

Currently, the API does not require authentication for local/development use. For production deployment:

**Bearer Token (Recommended):**
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!"}'
```

---

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t miyraa-api .

# Run container
docker run -p 8000:8000 miyraa-api

# Using docker-compose
docker-compose up -d
```

### Environment Variables
- `MODEL_PATH`: Path to PyTorch checkpoint (default: `/app/outputs/production_checkpoint.pt`)
- `ONNX_MODEL_PATH`: Path to ONNX model (default: `/app/outputs/xlm-roberta.quant.onnx`)
- `LOG_LEVEL`: Logging level (default: `info`)
- `MAX_WORKERS`: Number of worker processes (default: `4`)

---

## Performance Tips

1. **Use batch endpoint** for multiple texts (up to 100x faster than individual requests)
2. **Disable PII scrubbing** if not needed to reduce latency by ~30%
3. **Use ONNX models** for 2-3x faster inference on CPU
4. **Cache results** for frequently analyzed texts
5. **Use quantized models** for 4x smaller size with minimal accuracy loss

---

## Client Libraries

### Python
```bash
pip install requests
# See examples above
```

### JavaScript/TypeScript
```bash
npm install axios
```
```typescript
import axios from 'axios';

const response = await axios.post('http://localhost:8000/api/v1/analyze', {
  text: 'I love this!',
  return_vad: true
});
```

### cURL (Command Line)
All examples provided above work with standard cURL.

---

## Support

- **Repository:** https://github.com/kuruvamunirangadu/miyraa
- **Issues:** https://github.com/kuruvamunirangadu/miyraa/issues
- **Documentation:** See README.md in repository

---

## License

MIT License - See LICENSE file for details
