from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

from src.nlp.inference import get_engine
from src.nlp.safety.pii_scrub import scrub_pii, hash_id


class TextIn(BaseModel):
    text: str
    client_id: Optional[str] = None


app = FastAPI()

# simple in-memory rate limiter (per hashed client id or IP)
_RATE_STATE = {}
_RATE_LIMIT = 60  # requests
_RATE_WINDOW = 60  # seconds


@app.on_event("startup")
def startup_event():
    # warm the engine in background
    engine = get_engine()
    try:
        # best-effort: call predict on empty string to initialize weights/tokenizers
        engine.predict("")
    except Exception:
        pass


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


@app.post("/nlp/emotion/fingerprint")
def fingerprint(payload: TextIn, request: Request):
    # scrub PII and log only hashed client ids
    client_ip = request.client.host if request.client else "unknown"
    key_source = payload.client_id or client_ip
    key = hash_id(key_source)
    allowed = _check_rate(key)
    if not allowed:
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    text, pii_map = scrub_pii(payload.text)
    engine = get_engine()
    result = engine.predict(text[:128])  # cap text length for latency
    # attach hashed mapping (do not include raw PII)
    result["pii_hashes"] = pii_map
    result["client_hash"] = key
    return result
