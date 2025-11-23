# Deployment Quick Reference

## 🚀 Quick Start Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally
python -m uvicorn src.api.main:app --reload --port 8000

# Test health
curl http://localhost:8000/health
```

### Docker Single Container
```bash
# Build
docker build -t miyraa:latest .

# Run
docker run -d -p 8000:8000 --name miyraa-api miyraa:latest

# Logs
docker logs -f miyraa-api

# Stop
docker stop miyraa-api && docker rm miyraa-api
```

### Docker Compose
```bash
# Setup
cp .env.template .env

# Start
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

### With Monitoring
```bash
# Start full stack
docker-compose --profile monitoring up -d

# Access
open http://localhost:8000      # API
open http://localhost:9090      # Prometheus
open http://localhost:3000      # Grafana (admin/admin)
```

---

## 📊 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/nlp/emotion/fingerprint` | POST | Emotion detection |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |

---

## 🔧 Configuration (.env)

```bash
# Server
PORT=8000
WORKERS=1
LOG_LEVEL=info
LOG_FORMAT=json

# Model
MODEL_ID=xlm-roberta-base
WARMUP_TIMEOUT=30.0

# Features
PII_ANONYMIZE=true
SAFETY_THRESHOLD=0.5
RATE_LIMIT=60
```

---

## 📈 Key Metrics

```promql
# Request rate
rate(miyraa_requests_total[5m])

# Error rate
rate(miyraa_requests_total{status="500"}[5m])

# P95 latency
histogram_quantile(0.95, rate(miyraa_request_duration_seconds_bucket[5m]))

# Model health
miyraa_model_ready
```

---

## 🐛 Troubleshooting

### Service not starting
```bash
docker logs miyraa-api
docker-compose ps
```

### Health check failing
```bash
curl http://localhost:8000/health
docker exec miyraa-api curl http://localhost:8000/health
```

### View metrics
```bash
curl http://localhost:8000/metrics
```

---

## 📦 Image Sizes

- **Before**: ~2.5 GB
- **After**: ~1.2 GB (52% smaller)

---

## ⚙️ Resource Limits

```yaml
limits:
  cpus: '2'
  memory: 4G
reservations:
  cpus: '1'
  memory: 2G
```

---

## 📝 Makefile Commands

```bash
make help            # Show all commands
make docker-build    # Build image
make docker-run      # Run container
make compose-up      # Start compose stack
make monitoring      # Start with monitoring
make health-check    # Test health
```

---

## 🔐 Security Checklist

- [x] Non-root user
- [x] Minimal dependencies
- [x] Environment variables
- [ ] TLS/HTTPS
- [ ] Secrets management
- [ ] Image scanning

---

## 📚 Documentation

- [Deployment Guide](./DEPLOYMENT_GUIDE.md) - Full guide
- [Deployment Summary](./DEPLOYMENT_SUMMARY.md) - Implementation details
- [PII & Safety Guide](./PII_AND_SAFETY_GUIDE.md) - PII detection
- [Evaluation Guide](./EVALUATION_GUIDE.md) - Model evaluation

---

**Quick Links**:
- API Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
