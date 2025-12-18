from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import logging
import json
import sys
import os

from src.nlp.inference import get_engine
from src.nlp.safety.pii_scrub import scrub_pii, hash_id
from src.api.advanced import router as advanced_router


# Configure structured logging (JSON format for production)
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("miyraa")
logger.setLevel(LOG_LEVEL)

handler = logging.StreamHandler(sys.stdout)
if os.getenv("LOG_FORMAT", "json") == "json":
    handler.setFormatter(JSONFormatter())
else:
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
logger.addHandler(handler)


# Prometheus metrics
REQUEST_COUNT = Counter(
    'miyraa_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'miyraa_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)
RATE_LIMIT_EXCEEDED = Counter(
    'miyraa_rate_limit_exceeded_total',
    'Total number of rate limit violations'
)
PII_DETECTED = Counter(
    'miyraa_pii_detected_total',
    'Total number of PII entities detected',
    ['entity_type']
)
MODEL_READY = Gauge(
    'miyraa_model_ready',
    'Whether the model is ready (1) or not (0)'
)


class TextIn(BaseModel):
    text: str
    client_id: Optional[str] = None


app = FastAPI(
    title="Miyraa NLP API",
    description="Emotion detection and safety analysis API",
    version="1.0.0"
)


def load_model():
    """Backward-compatible shim for legacy tests expecting load_model."""
    return get_engine()

# Include advanced features router
app.include_router(advanced_router, prefix="/api/v1", tags=["advanced"])

# simple in-memory rate limiter (per hashed client id or IP)
_RATE_STATE = {}
_RATE_LIMIT = 60  # requests
_RATE_WINDOW = 60  # seconds

# Health check state
_HEALTH_STATE = {
    "model_loaded": False,
    "startup_time": time.time()
}


def _strip_text_fields(payload):
    """Remove raw text fields from payload recursively."""
    if isinstance(payload, dict):
        for key in ("text", "raw_text", "processed_text"):
            if key in payload:
                payload.pop(key)
        for key, value in list(payload.items()):
            payload[key] = _strip_text_fields(value)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            payload[index] = _strip_text_fields(item)
    return payload


@app.on_event("startup")
async def startup_event():
    """Warm the engine and initialize services"""
    logger.info("Starting Miyraa API service")
    engine = get_engine()
    try:
        # best-effort: call predict on empty string to initialize weights/tokenizers
        engine.predict("")
        _HEALTH_STATE["model_loaded"] = True
        MODEL_READY.set(1)
        logger.info("Model warmed up successfully")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}", exc_info=True)
        _HEALTH_STATE["model_loaded"] = False
        MODEL_READY.set(0)


def _check_rate(key: str):
    now = time.time()
    state = _RATE_STATE.get(key)
    if state is None or state[1] < now:
        _RATE_STATE[key] = [1, now + _RATE_WINDOW]
        return True
    if state[0] >= _RATE_LIMIT:
        return False
    state[0] += 1
    return True


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns 200 if service is healthy, 503 if not ready.
    """
    if not _HEALTH_STATE["model_loaded"]:
        logger.warning("Health check failed: model not loaded")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "uptime": time.time() - _HEALTH_STATE["startup_time"]
            }
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "uptime": time.time() - _HEALTH_STATE["startup_time"]
    }


@app.get("/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes/container orchestration.
    Returns 200 when service is ready to accept traffic.
    """
    if not _HEALTH_STATE["model_loaded"]:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "model not loaded"}
        )
    
    return {"ready": True}


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/nlp/emotion/fingerprint")
async def fingerprint(payload: TextIn, request: Request):
    """
    Emotion detection endpoint with PII scrubbing and rate limiting.
    """
    start_time = time.time()
    
    try:
        # scrub PII and log only hashed client ids
        client_ip = request.client.host if request.client else "unknown"
        key_source = payload.client_id or client_ip
        key = hash_id(key_source)
        
        logger.info(f"Request from client_hash={key}, text_length={len(payload.text)}")
        
        allowed = _check_rate(key)
        if not allowed:
            RATE_LIMIT_EXCEEDED.inc()
            REQUEST_COUNT.labels(endpoint="fingerprint", status="429").inc()
            logger.warning(f"Rate limit exceeded for client_hash={key}")
            raise HTTPException(status_code=429, detail="rate limit exceeded")

        text, pii_map = scrub_pii(payload.text)
        
        # Track PII detection
        if pii_map:
            for entity_type in pii_map.values():
                PII_DETECTED.labels(entity_type=entity_type).inc()
        
        engine = get_engine()
        result = engine.predict(text[:128])  # cap text length for latency
        result = _strip_text_fields(result)
        
        # attach hashed mapping (do not include raw PII)
        result["pii_hashes"] = pii_map
        result["client_hash"] = key
        
        # Track metrics
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint="fingerprint").observe(duration)
        REQUEST_COUNT.labels(endpoint="fingerprint", status="200").inc()
        
        logger.info(f"Request completed: client_hash={key}, duration={duration:.3f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="fingerprint", status="500").inc()
        logger.error(f"Request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="internal server error")
