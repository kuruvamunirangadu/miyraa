import re
import hashlib
from typing import Tuple

# Extended PII patterns: emails, phone numbers, URLs, handles, IBAN-like, short id numbers
PII_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", re.I),
    "phone": re.compile(r"\+?\d[\d\-\s]{7,}\d"),
    "url": re.compile(r"https?://[\w\.-/\?=&%#]+", re.I),
    "handle": re.compile(r"@\w{1,30}"),
    # rough IBAN / account patterns (very permissive)
    "iban": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}\b"),
    # Aadhaar-like 12 digits (India) or PAN-like patterns can be added
    "india_aadhaar": re.compile(r"\b\d{12}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

REPLACEMENTS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "url": "[URL]",
    "handle": "[HANDLE]",
    "iban": "[IBAN]",
    "india_aadhaar": "[ID]",
    "ssn_like": "[ID]",
}


def hash_id(s: str) -> str:
    """Return a stable sha256 hex digest for logging identifiers."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def scrub_pii(text: str) -> Tuple[str, dict]:
    """Replace PII substrings with placeholders and return a map of hashed values.

    Returns (scrubbed_text, {placeholder: hashed_value})
    """
    if not text:
        return text, {}
    mapping = {}
    out = text
    for key, pat in PII_PATTERNS.items():
        def _repl(m):
            val = m.group(0)
            h = hash_id(val)
            placeholder = REPLACEMENTS.get(key, "[PII]")
            # ensure uniqueness if multiple same-type matches
            tag = f"{placeholder}_{h[:8]}"
            mapping[tag] = h
            return tag

        out = pat.sub(_repl, out)
    return out, mapping


if __name__ == "__main__":
    s = "Contact me at foo.bar@example.com or +91 98765-43210. Visit https://example.com/user/@john_doe"
    scrubbed, mp = scrub_pii(s)
    print(scrubbed)
    print(mp)
