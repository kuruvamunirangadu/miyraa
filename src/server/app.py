from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import os
from pathlib import Path

# Globals filled at startup
TOKENIZER = None
MODEL = None
MODEL_ID = os.environ.get("MODEL_ID", "xlm-roberta-base")
WARMUP_TIMEOUT = float(os.environ.get("WARMUP_TIMEOUT", "30.0"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "5.0"))

app = FastAPI(title="Miyraa inference service")


async def _load_model_in_thread(model_id: str):
    """Load tokenizer and model in a thread to avoid blocking the event loop."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    def _sync_load():
        tok = AutoTokenizer.from_pretrained(model_id)
        m = AutoModelForSequenceClassification.from_pretrained(model_id)
        m.eval()
        return tok, m

    tok, m = await asyncio.to_thread(_sync_load)
    return tok, m


async def _preload_model():
    global TOKENIZER, MODEL
    try:
        tok, m = await asyncio.wait_for(_load_model_in_thread(MODEL_ID), timeout=WARMUP_TIMEOUT)
        TOKENIZER = tok
        MODEL = m
        app.state.warmed = True
        print(f"Model {MODEL_ID} preloaded and ready")
    except asyncio.TimeoutError:
        app.state.warmed = False
        print(f"Model preload timed out after {WARMUP_TIMEOUT}s")
    except Exception as e:
        app.state.warmed = False
        print(f"Model preload failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Kick off preload in background so first API call won't pay full load cost."""
    app.state.warmed = False
    # Start preload as background task but don't await here
    asyncio.create_task(_preload_model())


@app.get("/health")
async def health():
    return {"status": "ok", "warmed": bool(getattr(app.state, "warmed", False))}


@app.post("/predict")
async def predict(payload: dict):
    """A thin predict wrapper that ensures the model is loaded and enforces a per-request timeout.

    The payload is expected to be {"text": "..."} for this tiny example.
    """
    # allow assignment to module globals
    global TOKENIZER, MODEL

    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="missing 'text' in payload")

    # Ensure model is loaded; if not, try to load synchronously with a short timeout
    if TOKENIZER is None or MODEL is None:
        # attempt quick synchronous load with a short timeout to avoid blocking forever
        try:
            # await the same loader but with a tighter timeout
            tok, m = await asyncio.wait_for(_load_model_in_thread(MODEL_ID), timeout=REQUEST_TIMEOUT)
            # assign globals
            TOKENIZER, MODEL = tok, m
            app.state.warmed = True
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Model not warmed and loading timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    async def _run_inference(text: str):
        # Synchronous transformers inference in thread
        def _sync_infer():
            import torch
            inputs = TOKENIZER([text], return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                out = MODEL(**inputs, return_dict=True)
                logits = out.logits
                probs = logits.softmax(-1).tolist()[0]
                return {"probs": probs}

        return await asyncio.to_thread(_sync_infer)

    try:
        result = await asyncio.wait_for(_run_inference(text), timeout=REQUEST_TIMEOUT)
        return JSONResponse(result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


if __name__ == "__main__":
    # Simple launcher for local runs. Use uvicorn in production.
    import uvicorn

    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)
