# Expose nested packages so `src.api` and `src.nlp` resolve during imports
from . import api  # noqa: F401
from . import nlp  # noqa: F401

__all__ = ["nlp", "api"]
