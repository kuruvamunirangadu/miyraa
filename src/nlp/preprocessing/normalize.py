import re


def normalize_text(text: str, lower: bool = True, strip: bool = True) -> str:
    """Very small normalization helper used in tests.

    - collapses whitespace
    - optionally lowercases
    - strips surrounding whitespace
    """
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", text)
    if strip:
        s = s.strip()
    if lower:
        s = s.lower()
    return s


__all__ = ["normalize_text"]
