from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.nlp.safety.pii_detection import anonymize_for_inference
    from src.nlp.safety.safety_scoring import SafetyScorer, SafetyScore
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False


class DummyEngine:
    """
    Dummy inference engine for testing.
    
    Features:
    - Basic emotion/safety predictions
    - Optional PII anonymization
    - Safety risk scoring
    """
    
    def __init__(
        self,
        anonymize_pii: bool = False,
        pii_min_confidence: float = 0.7,
        safety_threshold: float = 0.35
    ):
        """
        Initialize dummy engine.
        
        Args:
            anonymize_pii: If True, anonymize PII before inference
            pii_min_confidence: Minimum confidence for PII detection
            safety_threshold: Safety classification threshold
        """
        self.anonymize_pii = anonymize_pii
        self.pii_min_confidence = pii_min_confidence
        
        # Initialize safety scorer
        self.safety_scorer = None
        if PII_AVAILABLE:
            self.safety_scorer = SafetyScorer(threshold=safety_threshold)
    
    def _preprocess_text(
        self,
        text: str
    ) -> Tuple[str, Optional[Dict]]:
        """
        Preprocess text with optional PII anonymization.
        
        Args:
            text: Input text
        
        Returns:
            (processed_text, pii_info)
        """
        if not self.anonymize_pii or not PII_AVAILABLE:
            return text, None
        
        # Anonymize PII
        anonymized_text, entities = anonymize_for_inference(
            text,
            min_confidence=self.pii_min_confidence
        )
        
        pii_info = {
            "pii_detected": len(entities) > 0,
            "pii_count": len(entities),
            "pii_types": list(set(e.entity_type.value for e in entities))
        }
        
        return anonymized_text, pii_info
    
    def _compute_safety_score(
        self,
        safety_blocked: bool
    ) -> Optional[Dict]:
        """
        Compute safety score from prediction.
        
        Args:
            safety_blocked: Boolean safety prediction
        
        Returns:
            Safety score dictionary or None
        """
        if not self.safety_scorer:
            return None
        
        # Simulate safety logits based on blocked status
        try:
            import torch
            if safety_blocked:
                logits = torch.tensor([-1.5, 2.0])  # [safe, unsafe]
            else:
                logits = torch.tensor([2.0, -1.5])
            
            score = self.safety_scorer.score_from_logits(logits)
            
            return {
                "is_safe": score.is_safe,
                "risk_index": score.risk_index,
                "risk_level": score.risk_level.value,
                "confidence": score.confidence,
                "explanation": score.explanation
            }
        except ImportError:
            # Torch not available, return basic score
            return {
                "is_safe": not safety_blocked,
                "risk_index": 0.8 if safety_blocked else 0.2,
                "risk_level": "high" if safety_blocked else "safe",
                "confidence": 0.85,
                "explanation": f"Content appears {'unsafe' if safety_blocked else 'safe'}"
            }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on text.
        
        Args:
            text: Input text
        
        Returns:
            Prediction dictionary with emotions, VAD, safety, etc.
        """
        # Preprocess with optional PII anonymization
        processed_text, pii_info = self._preprocess_text(text)
        
        # Return a deterministic-ish fingerprint for testing
        result = {
            "text": text,  # Original text
            "processed_text": processed_text,  # Anonymized if enabled
            "embed": [0.0, 0.1, 0.2],
            "vad": {"v": 0.5, "a": 0.5, "d": 0.5},
            "emotions": {"joy": 0.1, "anger": 0.0},
            "safety": {"blocked": False},
        }
        
        # Add PII info if available
        if pii_info:
            result["pii_info"] = pii_info
        
        # Add safety score if available
        safety_score = self._compute_safety_score(result["safety"]["blocked"])
        if safety_score:
            result["safety"]["score"] = safety_score
        
        return result


_ENGINE = None


def get_engine(
    anonymize_pii: bool = False,
    pii_min_confidence: float = 0.7,
    safety_threshold: float = 0.35
):
    """
    Get or create dummy engine instance.
    
    Args:
        anonymize_pii: Enable PII anonymization
        pii_min_confidence: PII detection confidence threshold
        safety_threshold: Safety classification threshold
    
    Returns:
        DummyEngine instance
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DummyEngine(
            anonymize_pii=anonymize_pii,
            pii_min_confidence=pii_min_confidence,
            safety_threshold=safety_threshold
        )
    return _ENGINE
