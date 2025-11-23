import re
import hashlib
from typing import Tuple, Dict, List, Optional

# Try to import Presidio for enterprise-grade PII detection
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    AnalyzerEngine = None
    AnonymizerEngine = None

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

# Presidio entity mapping to placeholders
PRESIDIO_ENTITY_MAP = {
    "PERSON": "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "PHONE_NUMBER": "[PHONE]",
    "CREDIT_CARD": "[CC]",
    "IBAN_CODE": "[IBAN]",
    "IP_ADDRESS": "[IP]",
    "LOCATION": "[LOCATION]",
    "DATE_TIME": "[DATE]",
    "NRP": "[NATIONALITY]",
    "URL": "[URL]",
    "US_SSN": "[SSN]",
    "US_PASSPORT": "[PASSPORT]",
    "US_DRIVER_LICENSE": "[LICENSE]",
    "MEDICAL_LICENSE": "[MEDICAL_ID]",
    "CRYPTO": "[CRYPTO]",
}

# Lazy-loaded Presidio engines
_analyzer_engine: Optional[AnalyzerEngine] = None
_anonymizer_engine: Optional[AnonymizerEngine] = None


def hash_id(s: str) -> str:
    """Return a stable sha256 hex digest for logging identifiers."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _get_presidio_engines():
    """Lazy initialization of Presidio engines."""
    global _analyzer_engine, _anonymizer_engine
    if not PRESIDIO_AVAILABLE:
        return None, None
    
    if _analyzer_engine is None:
        # Use spaCy small English model (fast)
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        _analyzer_engine = AnalyzerEngine(nlp_engine=provider.create_engine())
    
    if _anonymizer_engine is None:
        _anonymizer_engine = AnonymizerEngine()
    
    return _analyzer_engine, _anonymizer_engine


def scrub_pii_presidio(text: str, language: str = "en") -> Tuple[str, List[Dict]]:
    """Use Presidio to detect and anonymize PII with enterprise-grade accuracy.
    
    Args:
        text: Input text to scrub
        language: Language code (default: "en")
    
    Returns:
        (anonymized_text, list of detected entities with metadata)
    """
    if not text or not PRESIDIO_AVAILABLE:
        # Fallback to regex-based scrubbing
        return scrub_pii_regex(text)
    
    analyzer, anonymizer = _get_presidio_engines()
    if analyzer is None or anonymizer is None:
        return scrub_pii_regex(text)
    
    # Analyze text for PII entities
    results = analyzer.analyze(text=text, language=language)
    
    # Create anonymization operators for each entity type
    operators = {
        entity_type: {"type": "replace", "new_value": placeholder}
        for entity_type, placeholder in PRESIDIO_ENTITY_MAP.items()
    }
    
    # Anonymize detected entities
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )
    
    # Build entity metadata
    entities = [
        {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "score": result.score,
            "text_hash": hash_id(text[result.start:result.end])
        }
        for result in results
    ]
    
    return anonymized.text, entities


def scrub_pii_regex(text: str) -> Tuple[str, dict]:
    """Regex-based PII scrubbing (fallback when Presidio unavailable).
    
    Replace PII substrings with placeholders and return a map of hashed values.

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


def scrub_pii(text: str, use_presidio: bool = True) -> Tuple[str, dict]:
    """Main PII scrubbing function with automatic fallback.
    
    Args:
        text: Input text to scrub
        use_presidio: Whether to use Presidio (default: True if available)
    
    Returns:
        (scrubbed_text, metadata_dict)
    """
    if use_presidio and PRESIDIO_AVAILABLE:
        scrubbed, entities = scrub_pii_presidio(text)
        # Convert entities list to dict for backward compatibility
        metadata = {"method": "presidio", "entities": entities}
        return scrubbed, metadata
    else:
        scrubbed, mapping = scrub_pii_regex(text)
        metadata = {"method": "regex", "mappings": mapping}
        return scrubbed, metadata


if __name__ == "__main__":
    s = "Contact John Smith at foo.bar@example.com or +91 98765-43210. Visit https://example.com/user/@john_doe. SSN: 123-45-6789"
    
    print("=== Regex-based scrubbing ===")
    scrubbed_regex, meta_regex = scrub_pii(s, use_presidio=False)
    print(scrubbed_regex)
    print(meta_regex)
    
    if PRESIDIO_AVAILABLE:
        print("\n=== Presidio-based scrubbing ===")
        scrubbed_presidio, meta_presidio = scrub_pii(s, use_presidio=True)
        print(scrubbed_presidio)
        print(meta_presidio)
    else:
        print("\n=== Presidio not available, install with: pip install presidio-analyzer presidio-anonymizer spacy ===")
        print("=== Then download spaCy model: python -m spacy download en_core_web_sm ===")
