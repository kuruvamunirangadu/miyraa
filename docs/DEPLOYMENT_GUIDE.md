# Deployment Guide - Miyraa NLP API

**Version**: 1.0  
**Date**: November 2024  
**Status**: Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring](#monitoring)
7. [Health Checks](#health-checks)
8. [Logging](#logging)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Miyraa API is production-ready with:

✅ **Optimized Docker Image**:
- Multi-stage build for minimal size
- Non-root user for security
- Layer caching for fast rebuilds

✅ **Health & Readiness Probes**:
- `/health` - Container health check
- `/ready` - Kubernetes readiness probe
- `/metrics` - Prometheus metrics endpoint

✅ **Production Logging**:
- Structured JSON logs
- Configurable log levels
- Request tracing

✅ **Monitoring**:
- Prometheus metrics
- Grafana dashboards
- Request/error tracking

---

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m uvicorn src.api.main:app --reload --port 8000
```

### 2. Docker (Single Container)

```bash
# Build image
docker build -t miyraa:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e LOG_LEVEL=info \
  -e LOG_FORMAT=json \
  --name miyraa-api \
  miyraa:latest

# Check health
curl http://localhost:8000/health
```

### 3. Docker Compose (Full Stack)

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your configuration
nano .env

# Start all services
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Check logs
docker-compose logs -f miyraa-api

# Stop services
docker-compose down
```

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.template .env
```

#### Server Configuration

```bash
PORT=8000                    # API server port
WORKERS=1                    # Number of Uvicorn workers
LOG_LEVEL=info               # Logging level (debug, info, warning, error)
LOG_FORMAT=json              # Log format (json or text)
```

#### Model Configuration

```bash
MODEL_ID=xlm-roberta-base    # HuggingFace model ID
MODEL_PATH=/app/outputs/production_checkpoint.pt
ONNX_MODEL_PATH=/app/outputs/xlm-roberta.quant.onnx
WARMUP_TIMEOUT=30.0          # Model warmup timeout (seconds)
REQUEST_TIMEOUT=5.0          # Request timeout (seconds)
```

#### Rate Limiting

```bash
RATE_LIMIT=60                # Max requests per window
RATE_WINDOW=60               # Time window (seconds)
```

#### PII Detection

```bash
PII_ANONYMIZE=true           # Enable PII anonymization
PII_MIN_CONFIDENCE=0.7       # Minimum confidence threshold
```

#### Safety Scoring

```bash
SAFETY_THRESHOLD=0.5         # Safety classification threshold
```

#### Monitoring (Optional)

```bash
PROMETHEUS_PORT=9090         # Prometheus port
GRAFANA_PORT=3000            # Grafana port
GRAFANA_PASSWORD=admin       # Grafana admin password
```

---

## Docker Deployment

### Dockerfile Optimizations

The Dockerfile includes several optimizations:

1. **Multi-stage Build**:
   ```dockerfile
   FROM python:3.11-slim as builder
   # Build dependencies...
   
   FROM python:3.11-slim
   # Copy only necessary files
   ```

2. **Minimal Dependencies**:
   - Only `curl` installed for health checks
   - No build tools in final image

3. **Layer Caching**:
   - Requirements installed first
   - Application code copied last

4. **Security**:
   - Non-root user (`miyraa`)
   - No write permissions
   - Minimal attack surface

### Build Commands

```bash
# Build with default settings
docker build -t miyraa:latest .

# Build with custom tag
docker build -t miyraa:v1.0.0 .

# Build with build args
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t miyraa:latest .

# Multi-platform build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t miyraa:latest .
```

### Run Commands

```bash
# Basic run
docker run -d -p 8000:8000 miyraa:latest

# With environment variables
docker run -d \
  -p 8000:8000 \
  -e LOG_LEVEL=debug \
  -e WORKERS=2 \
  -e LOG_FORMAT=json \
  miyraa:latest

# With volume mounts
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/outputs:/app/outputs:ro \
  miyraa:latest

# With resource limits
docker run -d \
  -p 8000:8000 \
  --cpus="2" \
  --memory="4g" \
  miyraa:latest
```

---

## Kubernetes Deployment

### Deployment YAML

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: miyraa-api
  labels:
    app: miyraa
spec:
  replicas: 3
  selector:
    matchLabels:
      app: miyraa
  template:
    metadata:
      labels:
        app: miyraa
    spec:
      containers:
      - name: miyraa
        image: miyraa:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "info"
        - name: LOG_FORMAT
          value: "json"
        - name: WORKERS
          value: "1"
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      
      volumes:
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: miyraa-service
spec:
  selector:
    app: miyraa
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -l app=miyraa
kubectl get svc miyraa-service

# View logs
kubectl logs -l app=miyraa -f

# Scale deployment
kubectl scale deployment miyraa-api --replicas=5

# Update image
kubectl set image deployment/miyraa-api miyraa=miyraa:v1.0.1
```

---

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:

**Request Metrics**:
- `miyraa_requests_total` - Total requests by endpoint and status
- `miyraa_request_duration_seconds` - Request latency histogram

**Application Metrics**:
- `miyraa_model_ready` - Model readiness (1=ready, 0=not ready)
- `miyraa_rate_limit_exceeded_total` - Rate limit violations
- `miyraa_pii_detected_total` - PII entities detected by type

### Start Monitoring Stack

```bash
# Start Prometheus + Grafana
docker-compose --profile monitoring up -d

# Access Prometheus
open http://localhost:9090

# Access Grafana
open http://localhost:3000
# Username: admin
# Password: admin (or value from .env)
```

### Prometheus Queries

```promql
# Request rate
rate(miyraa_requests_total[5m])

# Error rate
rate(miyraa_requests_total{status="500"}[5m])

# P95 latency
histogram_quantile(0.95, rate(miyraa_request_duration_seconds_bucket[5m]))

# Rate limit violations
rate(miyraa_rate_limit_exceeded_total[5m])
```

### Grafana Dashboard

Import dashboard JSON from `monitoring/grafana-dashboard.json` (to be created).

Key panels:
- Request rate over time
- Error rate by endpoint
- Latency percentiles (P50, P95, P99)
- PII detection rate
- Model health status

---

## Health Checks

### Endpoints

#### `/health`

**Purpose**: Container health check  
**Returns**: 200 (healthy) or 503 (unhealthy)

```bash
curl http://localhost:8000/health
```

**Response (Healthy)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": 123.45
}
```

**Response (Unhealthy)**:
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "uptime": 10.5
}
```

#### `/ready`

**Purpose**: Kubernetes readiness probe  
**Returns**: 200 (ready) or 503 (not ready)

```bash
curl http://localhost:8000/ready
```

**Response (Ready)**:
```json
{
  "ready": true
}
```

**Response (Not Ready)**:
```json
{
  "ready": false,
  "reason": "model not loaded"
}
```

---

## Logging

### JSON Logs (Production)

Set `LOG_FORMAT=json` for structured logging:

```json
{
  "timestamp": "2024-11-23T10:30:45.123Z",
  "level": "INFO",
  "logger": "miyraa",
  "message": "Request completed",
  "module": "main",
  "function": "fingerprint",
  "line": 142,
  "client_hash": "abc123",
  "duration": 0.045
}
```

### Text Logs (Development)

Set `LOG_FORMAT=text` for human-readable logs:

```
2024-11-23 10:30:45,123 - miyraa - INFO - Request completed
```

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages with stack traces

### View Logs

```bash
# Docker
docker logs -f miyraa-api

# Docker Compose
docker-compose logs -f miyraa-api

# Kubernetes
kubectl logs -l app=miyraa -f

# Local file (if volume mounted)
tail -f logs/miyraa.log
```

### Log Rotation

Docker Compose includes automatic log rotation:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## Performance Tuning

### CPU Optimization

```bash
# Set number of workers
WORKERS=4  # Typically: num_cpus - 1

# Use ONNX quantized model
ONNX_MODEL_PATH=/app/outputs/xlm-roberta.quant.onnx
```

### Memory Optimization

```bash
# Reduce model size with quantization
# See docs/ONNX_QUANT_BENCH.md

# Set memory limits in Docker
docker run -m 4g miyraa:latest

# Set limits in Kubernetes
resources:
  limits:
    memory: "4Gi"
```

### Request Batching

For high throughput, consider batching:

```python
# In your client code
async def batch_predict(texts):
    tasks = [predict_one(text) for text in texts]
    return await asyncio.gather(*tasks)
```

### Caching

Add Redis for response caching:

```yaml
# docker-compose.yml
redis:
  image: redis:alpine
  ports:
    - "6379:6379"
```

---

## Troubleshooting

### Model Not Loading

**Symptom**: `/health` returns 503

**Solutions**:
1. Check model files exist:
   ```bash
   docker exec miyraa-api ls -l /app/outputs/
   ```

2. Increase warmup timeout:
   ```bash
   WARMUP_TIMEOUT=60.0
   ```

3. Check logs:
   ```bash
   docker logs miyraa-api | grep "Model"
   ```

### High Memory Usage

**Symptom**: Container OOM killed

**Solutions**:
1. Use quantized ONNX model (4x smaller)
2. Reduce number of workers
3. Increase memory limit
4. Profile with:
   ```bash
   docker stats miyraa-api
   ```

### Slow Requests

**Symptom**: High P95 latency

**Solutions**:
1. Check Prometheus metrics:
   ```promql
   histogram_quantile(0.95, miyraa_request_duration_seconds_bucket)
   ```

2. Enable debug logging:
   ```bash
   LOG_LEVEL=debug
   ```

3. Profile with:
   ```python
   import cProfile
   cProfile.run('engine.predict(text)')
   ```

### Rate Limiting Issues

**Symptom**: 429 errors

**Solutions**:
1. Increase rate limits:
   ```bash
   RATE_LIMIT=120
   RATE_WINDOW=60
   ```

2. Check metrics:
   ```bash
   curl http://localhost:8000/metrics | grep rate_limit
   ```

### PII Detection Errors

**Symptom**: Presidio not available

**Solutions**:
1. Install dependencies:
   ```bash
   pip install presidio-analyzer presidio-anonymizer spacy
   python -m spacy download en_core_web_sm
   ```

2. Disable Presidio:
   ```bash
   PII_ANONYMIZE=false
   ```

---

## Best Practices

### Security

✅ **DO**:
- Use non-root user
- Enable HTTPS/TLS in production
- Rotate secrets regularly
- Scan images for vulnerabilities
- Use secrets management (Vault, AWS Secrets Manager)

❌ **DON'T**:
- Run as root
- Commit secrets to git
- Use default passwords
- Expose internal ports

### Scaling

✅ **DO**:
- Use horizontal pod autoscaling (HPA)
- Set resource limits
- Use health checks
- Monitor metrics
- Use load balancing

❌ **DON'T**:
- Over-provision resources
- Skip health checks
- Ignore metrics
- Use single replica in production

### Monitoring

✅ **DO**:
- Set up alerts
- Monitor error rates
- Track latency percentiles
- Log structured data
- Use distributed tracing

❌ **DON'T**:
- Ignore warnings
- Skip log rotation
- Over-log sensitive data

---

## Production Checklist

Before deploying to production:

- [ ] Environment variables configured
- [ ] Secrets properly managed
- [ ] Health checks enabled
- [ ] Monitoring set up
- [ ] Log rotation configured
- [ ] Resource limits set
- [ ] TLS/HTTPS enabled
- [ ] Backup strategy defined
- [ ] Rollback plan documented
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Documentation updated

---

## Support

**Documentation**:
- [API Guide](./API_GUIDE.md)
- [PII & Safety Guide](./PII_AND_SAFETY_GUIDE.md)
- [Evaluation Guide](./EVALUATION_GUIDE.md)

**Issues**:
- GitHub Issues: https://github.com/kuruvamunirangadu/miyraa/issues

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Maintainer**: Miyraa Team
