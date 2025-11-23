"""
API endpoints for advanced features.
Integrates explainer, summarizer, multilang, and cache.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from ..nlp.inference.explainer import EmotionExplainer
from ..nlp.inference.summarizer import TextSummarizer
from ..nlp.inference.multilang import LanguageDetector, MultiLanguagePreprocessor
from ..nlp.inference.cache import get_cache, cached_inference

router = APIRouter()

# Initialize components
explainer = EmotionExplainer()
summarizer = TextSummarizer()
language_detector = LanguageDetector()
preprocessor = MultiLanguagePreprocessor()

# Initialize cache (can be configured via env vars)
cache_manager = get_cache(backend="lru", max_size=1000, ttl=3600)


# Models
class ExplainRequest(BaseModel):
    text: str
    emotion: str


class SummarizeRequest(BaseModel):
    text: str
    num_sentences: Optional[int] = None
    ratio: Optional[float] = None


class LanguageDetectRequest(BaseModel):
    text: str
    top_k: int = 3


# Endpoints
@router.get("/explain")
async def explain_emotion(
    text: str = Query(..., description="Input text"),
    emotion: str = Query(..., description="Emotion to explain")
):
    """
    Get explanation for emotion prediction.
    Returns keywords that contributed to the prediction.
    """
    try:
        result = explainer.explain(text, emotion)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_emotion_post(request: ExplainRequest):
    """
    Get explanation for emotion prediction (POST version).
    """
    try:
        result = explainer.explain(request.text, request.emotion)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explain/highlight")
async def explain_with_highlight(
    text: str = Query(..., description="Input text"),
    emotion: str = Query(..., description="Emotion to explain")
):
    """
    Get explanation with HTML-highlighted text.
    """
    try:
        explanation = explainer.explain(text, emotion)
        highlighted_html = explainer.highlight_text(text, explanation["keywords"])

        return {
            "explanation": explanation,
            "highlighted_html": highlighted_html
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summarize")
async def summarize_text(
    text: str = Query(..., description="Input text"),
    num_sentences: Optional[int] = Query(None, description="Number of sentences"),
    ratio: Optional[float] = Query(None, description="Compression ratio (0-1)")
):
    """
    Summarize text using extractive summarization.
    """
    try:
        if ratio is not None:
            result = summarizer.summarize_by_ratio(text, ratio)
        elif num_sentences is not None:
            result = summarizer.summarize(text, num_sentences)
        else:
            # Default: 3 sentences
            result = summarizer.summarize(text, 3)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize")
async def summarize_text_post(request: SummarizeRequest):
    """
    Summarize text (POST version).
    """
    try:
        if request.ratio is not None:
            result = summarizer.summarize_by_ratio(request.text, request.ratio)
        elif request.num_sentences is not None:
            result = summarizer.summarize(request.text, request.num_sentences)
        else:
            result = summarizer.summarize(request.text, 3)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/language/detect")
async def detect_language(
    text: str = Query(..., description="Input text"),
    top_k: int = Query(3, description="Top-k language candidates")
):
    """
    Detect language of text.
    """
    try:
        results = language_detector.detect(text, top_k)
        return {"text": text, "languages": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/language/detect")
async def detect_language_post(request: LanguageDetectRequest):
    """
    Detect language (POST version).
    """
    try:
        results = language_detector.detect(request.text, request.top_k)
        return {"text": request.text, "languages": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/language/preprocess")
async def preprocess_multilang(
    text: str = Query(..., description="Input text"),
    language: Optional[str] = Query(None, description="Language code (auto-detect if not provided)")
):
    """
    Preprocess text with language-specific rules.
    """
    try:
        result = preprocessor.preprocess(text, language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.
    """
    try:
        stats = cache_manager.stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cache entries.
    """
    try:
        cache_manager.clear()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
