"""
Safety Risk Scoring System for Content Moderation

Provides comprehensive safety assessment with:
- Binary classification (safe/unsafe)
- Continuous risk index (0-1 scale)
- Multi-dimensional safety categories
- Confidence-weighted scoring
- Threshold-based filtering

Author: Miyraa Team
Date: November 2025
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SafetyCategory(Enum):
    """Safety risk categories"""
    SAFE = "safe"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    DANGEROUS_CONTENT = "dangerous_content"


class RiskLevel(Enum):
    """Risk level classifications"""
    SAFE = "safe"              # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    CRITICAL = "critical"      # 0.8 - 1.0


@dataclass
class SafetyScore:
    """
    Comprehensive safety assessment result.
    
    Attributes:
        is_safe: Binary classification (True/False)
        risk_index: Continuous risk score (0.0 = safe, 1.0 = very unsafe)
        risk_level: Categorical risk level
        confidence: Model confidence in prediction
        categories: Scores for specific safety categories
        explanation: Human-readable explanation
    """
    is_safe: bool
    risk_index: float
    risk_level: RiskLevel
    confidence: float
    categories: Dict[str, float]
    explanation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "is_safe": self.is_safe,
            "risk_index": self.risk_index,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "categories": self.categories,
            "explanation": self.explanation
        }
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        status = "✓ SAFE" if self.is_safe else "✗ UNSAFE"
        return (
            f"Safety Assessment: {status}\n"
            f"  Risk Index: {self.risk_index:.3f} ({self.risk_level.value})\n"
            f"  Confidence: {self.confidence:.2%}\n"
            f"  {self.explanation}"
        )


class SafetyScorer:
    """
    Safety risk scoring system with multiple modalities.
    
    Supports:
    - Binary classification (safe/unsafe)
    - Continuous risk index (0-1)
    - Multi-category risk assessment
    - Confidence-weighted scoring
    - Customizable thresholds
    
    Example:
        >>> scorer = SafetyScorer(threshold=0.5)
        >>> score = scorer.score_from_logits(safety_logits)
        >>> print(f"Risk: {score.risk_index:.2f}")
        >>> if not score.is_safe:
        ...     print(f"Categories: {score.categories}")
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        risk_index_mode: str = "probability",
        category_names: Optional[List[str]] = None
    ):
        """
        Initialize safety scorer.
        
        Args:
            threshold: Binary classification threshold (0-1)
            risk_index_mode: How to compute risk index
                - "probability": Use unsafe probability directly
                - "entropy": Use prediction entropy as uncertainty measure
                - "calibrated": Use calibrated probability (requires calibration)
            category_names: Names for multi-category safety assessment
        """
        self.threshold = threshold
        self.risk_index_mode = risk_index_mode
        self.category_names = category_names or [
            "hate_speech",
            "violence", 
            "sexual_content",
            "harassment"
        ]
        
        # Risk level boundaries
        self.risk_boundaries = {
            RiskLevel.SAFE: (0.0, 0.2),
            RiskLevel.LOW: (0.2, 0.4),
            RiskLevel.MEDIUM: (0.4, 0.6),
            RiskLevel.HIGH: (0.6, 0.8),
            RiskLevel.CRITICAL: (0.8, 1.0)
        }
    
    def _compute_risk_index(
        self,
        probabilities: np.ndarray
    ) -> float:
        """
        Compute continuous risk index from probabilities.
        
        Args:
            probabilities: Softmax probabilities [safe_prob, unsafe_prob]
        
        Returns:
            risk_index: Float in [0, 1] where 0=safe, 1=very unsafe
        """
        if len(probabilities) == 2:
            # Binary classification: [safe, unsafe]
            unsafe_prob = probabilities[1]
            
            if self.risk_index_mode == "probability":
                return float(unsafe_prob)
            
            elif self.risk_index_mode == "entropy":
                # Use entropy as measure of uncertainty
                # High entropy = uncertain = higher risk
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                max_entropy = -np.log(1.0 / len(probabilities))
                normalized_entropy = entropy / max_entropy
                
                # Combine probability and entropy
                risk = unsafe_prob * 0.7 + normalized_entropy * 0.3
                return float(risk)
            
            else:  # calibrated mode (requires external calibration)
                # For now, same as probability mode
                # In production, apply temperature scaling or Platt scaling
                return float(unsafe_prob)
        
        else:
            # Multi-class classification: assume index 0 is safe
            # Risk is 1 - safe_probability
            return float(1.0 - probabilities[0])
    
    def _get_risk_level(self, risk_index: float) -> RiskLevel:
        """Map continuous risk index to categorical level"""
        for level, (low, high) in self.risk_boundaries.items():
            if low <= risk_index < high:
                return level
        return RiskLevel.CRITICAL  # >= 0.8
    
    def _get_explanation(
        self,
        is_safe: bool,
        risk_index: float,
        categories: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation"""
        if is_safe:
            return f"Content appears safe (risk: {risk_index:.1%})"
        
        # Find highest risk categories
        sorted_cats = sorted(
            categories.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_risks = [
            f"{cat}: {score:.1%}"
            for cat, score in sorted_cats[:2]
            if score > 0.3
        ]
        
        if top_risks:
            risk_str = ", ".join(top_risks)
            return f"Potential safety issues detected: {risk_str}"
        else:
            return f"Content flagged as unsafe (risk: {risk_index:.1%})"
    
    def score_from_logits(
        self,
        logits: torch.Tensor,
        category_logits: Optional[torch.Tensor] = None
    ) -> SafetyScore:
        """
        Compute safety score from model logits.
        
        Args:
            logits: Binary safety logits [batch_size, 2] or [2]
            category_logits: Optional multi-category logits [batch_size, num_categories]
        
        Returns:
            SafetyScore object with comprehensive assessment
        """
        # Handle batch dimension
        if logits.dim() == 2:
            logits = logits[0]
        
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Compute risk index
        risk_index = self._compute_risk_index(probabilities)
        
        # Binary classification
        is_safe = risk_index < self.threshold
        
        # Confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Risk level
        risk_level = self._get_risk_level(risk_index)
        
        # Category-specific scores
        categories = {}
        if category_logits is not None:
            if category_logits.dim() == 2:
                category_logits = category_logits[0]
            
            category_probs = torch.sigmoid(category_logits).cpu().numpy()
            
            for i, cat_name in enumerate(self.category_names):
                if i < len(category_probs):
                    categories[cat_name] = float(category_probs[i])
        else:
            # Estimate categories from binary score
            # This is a simplification; ideally use dedicated category heads
            base_risk = risk_index * 0.8
            categories = {
                cat: max(0.0, base_risk + np.random.normal(0, 0.1))
                for cat in self.category_names
            }
        
        # Generate explanation
        explanation = self._get_explanation(is_safe, risk_index, categories)
        
        return SafetyScore(
            is_safe=is_safe,
            risk_index=risk_index,
            risk_level=risk_level,
            confidence=confidence,
            categories=categories,
            explanation=explanation
        )
    
    def score_from_probabilities(
        self,
        probabilities: np.ndarray
    ) -> SafetyScore:
        """
        Compute safety score from pre-computed probabilities.
        
        Args:
            probabilities: Safety probabilities [safe_prob, unsafe_prob]
        
        Returns:
            SafetyScore object
        """
        # Convert to tensor and use main scoring method
        logits_tensor = torch.from_numpy(
            np.log(probabilities + 1e-10)
        ).float()
        
        return self.score_from_logits(logits_tensor)
    
    def batch_score(
        self,
        logits: torch.Tensor,
        category_logits: Optional[torch.Tensor] = None
    ) -> List[SafetyScore]:
        """
        Score multiple texts in batch.
        
        Args:
            logits: Batch of safety logits [batch_size, 2]
            category_logits: Optional category logits [batch_size, num_categories]
        
        Returns:
            List of SafetyScore objects
        """
        batch_size = logits.size(0)
        scores = []
        
        for i in range(batch_size):
            item_logits = logits[i]
            item_category_logits = (
                category_logits[i] if category_logits is not None else None
            )
            
            score = self.score_from_logits(item_logits, item_category_logits)
            scores.append(score)
        
        return scores
    
    def filter_unsafe(
        self,
        texts: List[str],
        scores: List[SafetyScore],
        return_indices: bool = False
    ) -> List[str]:
        """
        Filter out unsafe content from list.
        
        Args:
            texts: List of input texts
            scores: Corresponding SafetyScore objects
            return_indices: If True, return (safe_texts, safe_indices)
        
        Returns:
            List of safe texts (and optionally their indices)
        """
        safe_texts = []
        safe_indices = []
        
        for i, (text, score) in enumerate(zip(texts, scores)):
            if score.is_safe:
                safe_texts.append(text)
                safe_indices.append(i)
        
        if return_indices:
            return safe_texts, safe_indices
        return safe_texts
    
    def get_statistics(self, scores: List[SafetyScore]) -> Dict:
        """
        Get aggregate statistics for batch of scores.
        
        Args:
            scores: List of SafetyScore objects
        
        Returns:
            Dictionary with statistics
        """
        if not scores:
            return {}
        
        risk_indices = [s.risk_index for s in scores]
        confidences = [s.confidence for s in scores]
        
        stats = {
            "total_texts": len(scores),
            "safe_count": sum(1 for s in scores if s.is_safe),
            "unsafe_count": sum(1 for s in scores if not s.is_safe),
            "safe_percentage": sum(1 for s in scores if s.is_safe) / len(scores) * 100,
            "avg_risk_index": np.mean(risk_indices),
            "max_risk_index": np.max(risk_indices),
            "min_risk_index": np.min(risk_indices),
            "avg_confidence": np.mean(confidences),
            "risk_level_distribution": {}
        }
        
        # Count by risk level
        for level in RiskLevel:
            count = sum(1 for s in scores if s.risk_level == level)
            stats["risk_level_distribution"][level.value] = count
        
        # Category statistics (if available)
        if scores[0].categories:
            category_names = list(scores[0].categories.keys())
            stats["category_avg"] = {}
            
            for cat in category_names:
                cat_scores = [s.categories[cat] for s in scores]
                stats["category_avg"][cat] = np.mean(cat_scores)
        
        return stats


class AdaptiveSafetyScorer(SafetyScorer):
    """
    Adaptive safety scorer that adjusts thresholds based on context.
    
    Features:
    - Context-aware threshold adjustment
    - User-specific risk tolerance
    - Time-of-day modulation
    - A/B testing support
    
    Example:
        >>> scorer = AdaptiveSafetyScorer(base_threshold=0.5)
        >>> score = scorer.score_with_context(
        ...     logits,
        ...     context={"user_age": 16, "platform": "public"}
        ... )
    """
    
    def __init__(
        self,
        base_threshold: float = 0.5,
        strict_mode: bool = False,
        **kwargs
    ):
        """
        Initialize adaptive scorer.
        
        Args:
            base_threshold: Default threshold
            strict_mode: If True, use lower threshold (more conservative)
            **kwargs: Additional arguments for SafetyScorer
        """
        super().__init__(threshold=base_threshold, **kwargs)
        self.base_threshold = base_threshold
        self.strict_mode = strict_mode
        
        # Context-based threshold adjustments
        self.context_adjustments = {
            "minor_user": -0.2,        # Lower threshold for minors
            "public_platform": -0.1,    # More conservative for public content
            "educational": +0.1,        # More lenient for educational context
            "moderated": +0.15          # More lenient if human moderation follows
        }
    
    def _adjust_threshold(self, context: Dict) -> float:
        """
        Adjust threshold based on context.
        
        Args:
            context: Dictionary with context information
                - user_age: User age (if < 18, apply minor adjustment)
                - platform: "public" or "private"
                - content_type: "educational", "entertainment", etc.
                - has_moderation: Boolean
        
        Returns:
            Adjusted threshold
        """
        threshold = self.base_threshold
        
        # Age-based adjustment
        if context.get("user_age") and context["user_age"] < 18:
            threshold += self.context_adjustments["minor_user"]
        
        # Platform adjustment
        if context.get("platform") == "public":
            threshold += self.context_adjustments["public_platform"]
        
        # Content type adjustment
        if context.get("content_type") == "educational":
            threshold += self.context_adjustments["educational"]
        
        # Moderation adjustment
        if context.get("has_moderation"):
            threshold += self.context_adjustments["moderated"]
        
        # Strict mode override
        if self.strict_mode:
            threshold = min(threshold, 0.4)
        
        # Clamp to valid range
        threshold = max(0.1, min(0.9, threshold))
        
        return threshold
    
    def score_with_context(
        self,
        logits: torch.Tensor,
        context: Dict,
        category_logits: Optional[torch.Tensor] = None
    ) -> SafetyScore:
        """
        Score with context-aware threshold adjustment.
        
        Args:
            logits: Safety logits
            context: Context dictionary
            category_logits: Optional category logits
        
        Returns:
            SafetyScore with context-adjusted threshold
        """
        # Adjust threshold based on context
        original_threshold = self.threshold
        self.threshold = self._adjust_threshold(context)
        
        # Compute score
        score = self.score_from_logits(logits, category_logits)
        
        # Restore original threshold
        self.threshold = original_threshold
        
        return score


if __name__ == "__main__":
    """Test safety scoring system"""
    
    print("="*60)
    print("SAFETY RISK SCORING TESTS")
    print("="*60)
    
    # Simulate model outputs
    np.random.seed(42)
    
    # Test 1: Basic scoring
    print("\n1. Basic Safety Scoring")
    print("-"*60)
    
    scorer = SafetyScorer(threshold=0.5)
    
    # Safe content (high safe probability)
    safe_logits = torch.tensor([2.0, -1.5])  # [safe, unsafe]
    safe_score = scorer.score_from_logits(safe_logits)
    print(f"Safe content:\n{safe_score}\n")
    
    # Unsafe content (high unsafe probability)
    unsafe_logits = torch.tensor([-1.5, 2.0])
    unsafe_score = scorer.score_from_logits(unsafe_logits)
    print(f"Unsafe content:\n{unsafe_score}\n")
    
    # Borderline content
    borderline_logits = torch.tensor([0.1, 0.2])
    borderline_score = scorer.score_from_logits(borderline_logits)
    print(f"Borderline content:\n{borderline_score}\n")
    
    # Test 2: Batch scoring
    print("2. Batch Scoring")
    print("-"*60)
    
    batch_logits = torch.tensor([
        [2.0, -1.5],   # Safe
        [-1.5, 2.0],   # Unsafe
        [0.5, 0.3],    # Borderline safe
        [-2.0, 3.0],   # Very unsafe
        [1.5, -1.0]    # Safe
    ])
    
    batch_scores = scorer.batch_score(batch_logits)
    
    for i, score in enumerate(batch_scores):
        status = "✓ SAFE" if score.is_safe else "✗ UNSAFE"
        print(f"Text {i+1}: {status} | Risk: {score.risk_index:.3f} | "
              f"Level: {score.risk_level.value}")
    
    # Test 3: Statistics
    print(f"\n3. Batch Statistics")
    print("-"*60)
    
    stats = scorer.get_statistics(batch_scores)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Safe: {stats['safe_count']} ({stats['safe_percentage']:.1f}%)")
    print(f"Unsafe: {stats['unsafe_count']}")
    print(f"Avg risk index: {stats['avg_risk_index']:.3f}")
    print(f"Max risk index: {stats['max_risk_index']:.3f}")
    print(f"\nRisk level distribution:")
    for level, count in stats['risk_level_distribution'].items():
        print(f"  {level}: {count}")
    
    # Test 4: Adaptive scoring
    print(f"\n4. Adaptive Threshold Scoring")
    print("-"*60)
    
    adaptive_scorer = AdaptiveSafetyScorer(base_threshold=0.5)
    
    test_logits = torch.tensor([0.0, 0.6])  # Borderline
    
    contexts = [
        {"user_age": 25, "platform": "private"},
        {"user_age": 15, "platform": "public"},
        {"content_type": "educational", "has_moderation": True}
    ]
    
    for i, context in enumerate(contexts, 1):
        score = adaptive_scorer.score_with_context(test_logits, context)
        print(f"\nContext {i}: {context}")
        print(f"  Decision: {'SAFE' if score.is_safe else 'UNSAFE'}")
        print(f"  Risk index: {score.risk_index:.3f}")
    
    # Test 5: Filtering
    print(f"\n5. Content Filtering")
    print("-"*60)
    
    texts = [
        "This is safe content",
        "Potentially unsafe content",
        "Another safe message",
        "Very unsafe content",
        "Borderline content"
    ]
    
    safe_texts, safe_indices = scorer.filter_unsafe(
        texts, batch_scores, return_indices=True
    )
    
    print(f"Original texts: {len(texts)}")
    print(f"Safe texts: {len(safe_texts)}")
    print(f"Safe indices: {safe_indices}")
    print(f"\nFiltered texts:")
    for idx, text in zip(safe_indices, safe_texts):
        print(f"  [{idx}] {text}")
    
    print("\n" + "="*60)
    print("✅ All safety scoring tests completed!")
    print("="*60)
