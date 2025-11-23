"""
Enhanced PII Detection with Confidence Scoring

Hybrid approach combining:
- Regex patterns (fast, high precision for common patterns)
- Presidio (NER-based, high recall for contextual PII)
- Confidence scoring for each detected entity
- Anonymization pipeline for pre-inference processing

Author: Miyraa Team
Date: November 2025
"""

import re
import hashlib
from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Try to import Presidio for enterprise-grade PII detection
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    AnalyzerEngine = None
    AnonymizerEngine = None
    RecognizerResult = None


class PIIEntityType(Enum):
    """Standardized PII entity types"""
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    HANDLE = "HANDLE"
    IBAN = "IBAN"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    PASSPORT = "PASSPORT"
    LICENSE = "LICENSE"
    IP_ADDRESS = "IP_ADDRESS"
    PERSON_NAME = "PERSON_NAME"
    LOCATION = "LOCATION"
    DATE_TIME = "DATE_TIME"
    MEDICAL_ID = "MEDICAL_ID"
    AADHAAR = "AADHAAR"  # India ID
    CRYPTO_WALLET = "CRYPTO_WALLET"
    UNKNOWN = "UNKNOWN"


@dataclass
class PIIEntity:
    """Detected PII entity with metadata"""
    entity_type: PIIEntityType
    start: int
    end: int
    text: str
    confidence: float
    detection_method: str  # "regex", "presidio", or "hybrid"
    text_hash: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "entity_type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "text_hash": self.text_hash
        }


# Enhanced PII regex patterns with confidence scores
PII_PATTERNS = {
    "EMAIL": {
        "pattern": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "confidence": 0.95,  # High confidence for valid email format
        "placeholder": "[EMAIL]"
    },
    "PHONE": {
        "pattern": re.compile(
            r"(\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}|\+?\d{10,15}"
        ),
        "confidence": 0.85,
        "placeholder": "[PHONE]"
    },
    "URL": {
        "pattern": re.compile(
            r"https?://(?:www\.)?[\w\-]+\.[\w\-./\?=&#%~]+",
            re.IGNORECASE
        ),
        "confidence": 0.95,
        "placeholder": "[URL]"
    },
    "HANDLE": {
        "pattern": re.compile(r"@\w{1,30}\b"),
        "confidence": 0.80,
        "placeholder": "[HANDLE]"
    },
    "IBAN": {
        "pattern": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
        "confidence": 0.75,
        "placeholder": "[IBAN]"
    },
    "SSN": {
        "pattern": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "confidence": 0.90,
        "placeholder": "[SSN]"
    },
    "AADHAAR": {
        "pattern": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
        "confidence": 0.70,
        "placeholder": "[ID]"
    },
    "IP_ADDRESS": {
        "pattern": re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        ),
        "confidence": 0.85,
        "placeholder": "[IP]"
    },
    "CREDIT_CARD": {
        "pattern": re.compile(
            r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
        ),
        "confidence": 0.65,  # Lower confidence, many false positives
        "placeholder": "[CC]"
    }
}


# Presidio entity type mapping
PRESIDIO_ENTITY_MAP = {
    "PERSON": PIIEntityType.PERSON_NAME,
    "EMAIL_ADDRESS": PIIEntityType.EMAIL,
    "PHONE_NUMBER": PIIEntityType.PHONE,
    "CREDIT_CARD": PIIEntityType.CREDIT_CARD,
    "IBAN_CODE": PIIEntityType.IBAN,
    "IP_ADDRESS": PIIEntityType.IP_ADDRESS,
    "LOCATION": PIIEntityType.LOCATION,
    "DATE_TIME": PIIEntityType.DATE_TIME,
    "NRP": PIIEntityType.LOCATION,
    "URL": PIIEntityType.URL,
    "US_SSN": PIIEntityType.SSN,
    "US_PASSPORT": PIIEntityType.PASSPORT,
    "US_DRIVER_LICENSE": PIIEntityType.LICENSE,
    "MEDICAL_LICENSE": PIIEntityType.MEDICAL_ID,
    "CRYPTO": PIIEntityType.CRYPTO_WALLET,
}


# Lazy-loaded Presidio engines
_analyzer_engine: Optional[AnalyzerEngine] = None
_anonymizer_engine: Optional[AnonymizerEngine] = None


def hash_text(text: str) -> str:
    """Generate stable SHA256 hash for PII text"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _get_presidio_engines():
    """Lazy initialization of Presidio engines"""
    global _analyzer_engine, _anonymizer_engine
    
    if not PRESIDIO_AVAILABLE:
        return None, None
    
    if _analyzer_engine is None:
        try:
            # Use spaCy small English model (fast)
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            })
            _analyzer_engine = AnalyzerEngine(nlp_engine=provider.create_engine())
        except Exception as e:
            print(f"Warning: Failed to initialize Presidio analyzer: {e}")
            return None, None
    
    if _anonymizer_engine is None:
        _anonymizer_engine = AnonymizerEngine()
    
    return _analyzer_engine, _anonymizer_engine


class PIIDetector:
    """
    Hybrid PII detector combining regex and Presidio.
    
    Features:
    - Regex-based detection for common patterns (fast, high precision)
    - Presidio NER-based detection for contextual PII (high recall)
    - Confidence scoring for each detected entity
    - Deduplication to avoid double-counting overlapping detections
    
    Example:
        >>> detector = PIIDetector()
        >>> entities = detector.detect_pii("Contact john@example.com or call 555-1234")
        >>> for entity in entities:
        ...     print(f"{entity.entity_type}: {entity.confidence:.2f}")
    """
    
    def __init__(
        self,
        use_presidio: bool = True,
        min_confidence: float = 0.5,
        language: str = "en"
    ):
        """
        Initialize PII detector.
        
        Args:
            use_presidio: Whether to use Presidio (if available)
            min_confidence: Minimum confidence threshold for detections
            language: Language code for Presidio analysis
        """
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        self.min_confidence = min_confidence
        self.language = language
        
        if self.use_presidio:
            self.analyzer, self.anonymizer = _get_presidio_engines()
            if self.analyzer is None:
                self.use_presidio = False
    
    def _detect_regex(self, text: str) -> List[PIIEntity]:
        """Detect PII using regex patterns"""
        entities = []
        
        for entity_type_str, config in PII_PATTERNS.items():
            pattern = config["pattern"]
            confidence = config["confidence"]
            
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                
                # Skip if below minimum confidence
                if confidence < self.min_confidence:
                    continue
                
                entity = PIIEntity(
                    entity_type=PIIEntityType[entity_type_str],
                    start=match.start(),
                    end=match.end(),
                    text=matched_text,
                    confidence=confidence,
                    detection_method="regex",
                    text_hash=hash_text(matched_text)
                )
                entities.append(entity)
        
        return entities
    
    def _detect_presidio(self, text: str) -> List[PIIEntity]:
        """Detect PII using Presidio NER"""
        if not self.use_presidio or self.analyzer is None:
            return []
        
        entities = []
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                score_threshold=self.min_confidence
            )
            
            for result in results:
                # Map Presidio entity type to our enum
                entity_type = PRESIDIO_ENTITY_MAP.get(
                    result.entity_type,
                    PIIEntityType.UNKNOWN
                )
                
                entity = PIIEntity(
                    entity_type=entity_type,
                    start=result.start,
                    end=result.end,
                    text=text[result.start:result.end],
                    confidence=result.score,
                    detection_method="presidio",
                    text_hash=hash_text(text[result.start:result.end])
                )
                entities.append(entity)
        
        except Exception as e:
            print(f"Warning: Presidio detection failed: {e}")
        
        return entities
    
    def _merge_overlapping(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Merge overlapping entity detections, keeping highest confidence.
        
        When both regex and Presidio detect the same entity:
        - If they agree on type: keep higher confidence
        - If they disagree: boost confidence via hybrid method
        """
        if not entities:
            return []
        
        # Sort by start position
        entities = sorted(entities, key=lambda e: (e.start, -e.confidence))
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # Check for overlap
            if entity.start < current.end:
                # Overlapping entities
                if entity.entity_type == current.entity_type:
                    # Same type: keep higher confidence
                    if entity.confidence > current.confidence:
                        current = entity
                else:
                    # Different types: hybrid detection boosts confidence
                    if entity.confidence > current.confidence:
                        entity.confidence = min(0.99, entity.confidence + 0.1)
                        entity.detection_method = "hybrid"
                        current = entity
                    else:
                        current.confidence = min(0.99, current.confidence + 0.1)
                        current.detection_method = "hybrid"
            else:
                # No overlap: add current and move to next
                merged.append(current)
                current = entity
        
        # Add last entity
        merged.append(current)
        
        return merged
    
    def detect_pii(self, text: str) -> List[PIIEntity]:
        """
        Detect all PII entities in text using hybrid approach.
        
        Args:
            text: Input text to analyze
        
        Returns:
            List of PIIEntity objects with confidence scores
        """
        if not text:
            return []
        
        # Detect with regex
        regex_entities = self._detect_regex(text)
        
        # Detect with Presidio (if enabled)
        presidio_entities = self._detect_presidio(text) if self.use_presidio else []
        
        # Merge and deduplicate
        all_entities = regex_entities + presidio_entities
        merged_entities = self._merge_overlapping(all_entities)
        
        return merged_entities
    
    def get_detection_stats(self, text: str) -> Dict:
        """
        Get statistics about PII detection.
        
        Returns:
            Dictionary with detection counts by type and method
        """
        entities = self.detect_pii(text)
        
        stats = {
            "total_entities": len(entities),
            "by_type": {},
            "by_method": {"regex": 0, "presidio": 0, "hybrid": 0},
            "avg_confidence": 0.0
        }
        
        if not entities:
            return stats
        
        # Count by type
        for entity in entities:
            entity_type = entity.entity_type.value
            stats["by_type"][entity_type] = stats["by_type"].get(entity_type, 0) + 1
            stats["by_method"][entity.detection_method] += 1
        
        # Average confidence
        stats["avg_confidence"] = sum(e.confidence for e in entities) / len(entities)
        
        return stats


class PIIAnonymizer:
    """
    Anonymize PII in text for safe inference.
    
    Features:
    - Replace detected PII with typed placeholders
    - Maintain reversible mapping (optional)
    - Support for anonymize-before-inference mode
    
    Example:
        >>> anonymizer = PIIAnonymizer()
        >>> result = anonymizer.anonymize("Email john@example.com")
        >>> print(result.text)  # "Email [EMAIL_1a2b3c]"
        >>> print(result.entities)  # [PIIEntity(...)]
    """
    
    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        include_hash: bool = True
    ):
        """
        Initialize anonymizer.
        
        Args:
            detector: PIIDetector instance (creates new if None)
            include_hash: Include hash suffix in placeholders for uniqueness
        """
        self.detector = detector or PIIDetector()
        self.include_hash = include_hash
    
    def anonymize(
        self,
        text: str,
        reversible: bool = False
    ) -> Tuple[str, List[PIIEntity], Optional[Dict[str, str]]]:
        """
        Anonymize PII in text.
        
        Args:
            text: Input text
            reversible: If True, return mapping to restore original text
        
        Returns:
            (anonymized_text, detected_entities, optional_reverse_map)
        """
        if not text:
            return text, [], None
        
        # Detect PII
        entities = self.detector.detect_pii(text)
        
        if not entities:
            return text, [], None
        
        # Sort by start position (reverse to replace from end)
        entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)
        
        # Build reverse mapping if requested
        reverse_map = {} if reversible else None
        
        # Replace entities with placeholders
        anonymized = text
        for entity in entities_sorted:
            # Generate placeholder
            placeholder = f"[{entity.entity_type.value}"
            if self.include_hash:
                placeholder += f"_{entity.text_hash}"
            placeholder += "]"
            
            # Store reverse mapping
            if reversible:
                reverse_map[placeholder] = entity.text
            
            # Replace in text
            anonymized = (
                anonymized[:entity.start] +
                placeholder +
                anonymized[entity.end:]
            )
        
        return anonymized, entities, reverse_map
    
    def deanonymize(
        self,
        text: str,
        reverse_map: Dict[str, str]
    ) -> str:
        """
        Restore original text from anonymized version.
        
        Args:
            text: Anonymized text
            reverse_map: Mapping from placeholders to original text
        
        Returns:
            Restored text
        """
        if not reverse_map:
            return text
        
        restored = text
        for placeholder, original in reverse_map.items():
            restored = restored.replace(placeholder, original)
        
        return restored


def anonymize_for_inference(
    text: str,
    min_confidence: float = 0.7,
    use_presidio: bool = True
) -> Tuple[str, List[Dict]]:
    """
    Convenience function to anonymize text before inference.
    
    This is useful when you want to ensure no PII reaches the model.
    
    Args:
        text: Input text
        min_confidence: Minimum confidence for PII detection
        use_presidio: Whether to use Presidio
    
    Returns:
        (anonymized_text, list of entity metadata dicts)
    
    Example:
        >>> text = "Contact john@example.com or call 555-1234"
        >>> anon_text, entities = anonymize_for_inference(text)
        >>> print(anon_text)  # "Contact [EMAIL_abc123] or call [PHONE_def456]"
    """
    detector = PIIDetector(
        use_presidio=use_presidio,
        min_confidence=min_confidence
    )
    anonymizer = PIIAnonymizer(detector=detector, include_hash=True)
    
    anonymized_text, entities, _ = anonymizer.anonymize(text, reversible=False)
    
    # Convert entities to dicts for JSON serialization
    entity_dicts = [entity.to_dict() for entity in entities]
    
    return anonymized_text, entity_dicts


# Maintain backward compatibility with old API
def scrub_pii(
    text: str,
    use_presidio: bool = True
) -> Tuple[str, Dict]:
    """
    Legacy API for PII scrubbing (backward compatible).
    
    Args:
        text: Input text
        use_presidio: Whether to use Presidio
    
    Returns:
        (scrubbed_text, metadata_dict)
    """
    anonymized_text, entities = anonymize_for_inference(
        text,
        use_presidio=use_presidio
    )
    
    metadata = {
        "method": "hybrid" if use_presidio and PRESIDIO_AVAILABLE else "regex",
        "entities": entities,
        "total_entities": len(entities)
    }
    
    return anonymized_text, metadata


if __name__ == "__main__":
    """Test PII detection and anonymization"""
    
    test_text = (
        "Contact John Smith at john.smith@example.com or call +1-555-123-4567. "
        "Visit https://example.com/@johndoe. "
        "SSN: 123-45-6789, IP: 192.168.1.1"
    )
    
    print("="*60)
    print("PII DETECTION & ANONYMIZATION TESTS")
    print("="*60)
    print(f"\nOriginal text:\n{test_text}\n")
    
    # Test 1: Detection with confidence scoring
    print("1. Hybrid PII Detection (Regex + Presidio)")
    print("-"*60)
    
    detector = PIIDetector(use_presidio=True, min_confidence=0.5)
    entities = detector.detect_pii(test_text)
    
    print(f"Detected {len(entities)} PII entities:\n")
    for entity in entities:
        print(f"  {entity.entity_type.value:15s} | "
              f"Confidence: {entity.confidence:.2f} | "
              f"Method: {entity.detection_method:8s} | "
              f"Text: {entity.text}")
    
    # Test 2: Detection statistics
    print(f"\n2. Detection Statistics")
    print("-"*60)
    
    stats = detector.get_detection_stats(test_text)
    print(f"Total entities: {stats['total_entities']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"\nBy type:")
    for entity_type, count in stats['by_type'].items():
        print(f"  {entity_type}: {count}")
    print(f"\nBy method:")
    for method, count in stats['by_method'].items():
        print(f"  {method}: {count}")
    
    # Test 3: Anonymization
    print(f"\n3. Anonymization (with hash)")
    print("-"*60)
    
    anonymizer = PIIAnonymizer(detector=detector, include_hash=True)
    anon_text, entities, reverse_map = anonymizer.anonymize(test_text, reversible=True)
    
    print(f"Anonymized text:\n{anon_text}\n")
    
    if reverse_map:
        print("Reverse mapping:")
        for placeholder, original in list(reverse_map.items())[:3]:
            print(f"  {placeholder} → {original}")
    
    # Test 4: Convenience function
    print(f"\n4. Anonymize for Inference (convenience API)")
    print("-"*60)
    
    anon_text, entity_dicts = anonymize_for_inference(test_text)
    print(f"Anonymized: {anon_text}\n")
    print(f"Detected {len(entity_dicts)} entities")
    
    # Test 5: Regex-only mode
    print(f"\n5. Regex-only Detection (fast mode)")
    print("-"*60)
    
    detector_regex = PIIDetector(use_presidio=False, min_confidence=0.5)
    entities_regex = detector_regex.detect_pii(test_text)
    
    print(f"Detected {len(entities_regex)} PII entities (regex only):\n")
    for entity in entities_regex:
        print(f"  {entity.entity_type.value:15s} | "
              f"Confidence: {entity.confidence:.2f}")
    
    print("\n" + "="*60)
    print("✅ All PII detection tests completed!")
    print("="*60)
    
    if not PRESIDIO_AVAILABLE:
        print("\n⚠️  Note: Presidio not available. Install with:")
        print("    pip install presidio-analyzer presidio-anonymizer spacy")
        print("    python -m spacy download en_core_web_sm")
