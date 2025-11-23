"""Miyraa Emotion Taxonomy and Label Mappings.

This module defines the canonical emotion taxonomy for Miyraa, including:
- Core emotions (11 primary categories)
- VAD (Valence-Arousal-Dominance) dimensions
- Safety categories
- Style and intent classes
- Mapping from GoEmotions to Miyraa taxonomy
"""

from typing import Dict, List, Set, Tuple
import json

# =============================================================================
# CORE EMOTION TAXONOMY (11 Classes)
# =============================================================================

EMOTIONS = [
    "joy",           # Happiness, delight, contentment
    "love",          # Affection, care, admiration
    "surprise",      # Shock, amazement, disbelief
    "sadness",       # Sorrow, grief, disappointment
    "anger",         # Rage, frustration, annoyance
    "fear",          # Anxiety, terror, worry
    "disgust",       # Revulsion, distaste, contempt
    "calm",          # Peace, serenity, relaxation
    "excitement",    # Enthusiasm, anticipation, eagerness
    "confusion",     # Uncertainty, bewilderment, puzzlement
    "neutral",       # No strong emotion, factual
]

# Emotion descriptions for clarity
EMOTION_DESCRIPTIONS = {
    "joy": "Positive emotion including happiness, delight, pleasure, contentment, satisfaction",
    "love": "Affection, care, admiration, appreciation, gratitude, romantic feelings",
    "surprise": "Shock, amazement, astonishment, disbelief, unexpectedness",
    "sadness": "Sorrow, grief, melancholy, disappointment, loneliness, regret",
    "anger": "Rage, fury, frustration, irritation, annoyance, resentment",
    "fear": "Anxiety, terror, worry, panic, nervousness, dread",
    "disgust": "Revulsion, distaste, contempt, repulsion, aversion",
    "calm": "Peace, serenity, tranquility, relaxation, composure",
    "excitement": "Enthusiasm, anticipation, eagerness, thrill, exhilaration",
    "confusion": "Uncertainty, bewilderment, puzzlement, perplexity, disorientation",
    "neutral": "No strong emotion, factual, objective, indifferent",
}

# =============================================================================
# GOEMOTIONS → MIYRAA MAPPING
# =============================================================================

# GoEmotions has 27 emotion categories + neutral
# We map them to our 11 core emotions with confidence weights

GOEMOTIONS_TO_MIYRAA = {
    # Direct mappings (high confidence)
    "admiration": [("love", 0.8), ("joy", 0.2)],
    "amusement": [("joy", 1.0)],
    "anger": [("anger", 1.0)],
    "annoyance": [("anger", 0.7), ("disgust", 0.3)],
    "approval": [("joy", 0.5), ("love", 0.5)],
    "caring": [("love", 1.0)],
    "confusion": [("confusion", 1.0)],
    "curiosity": [("excitement", 0.6), ("surprise", 0.4)],
    "desire": [("excitement", 0.7), ("love", 0.3)],
    "disappointment": [("sadness", 0.8), ("anger", 0.2)],
    "disapproval": [("anger", 0.5), ("disgust", 0.5)],
    "disgust": [("disgust", 1.0)],
    "embarrassment": [("sadness", 0.5), ("fear", 0.5)],
    "excitement": [("excitement", 1.0)],
    "fear": [("fear", 1.0)],
    "gratitude": [("love", 0.8), ("joy", 0.2)],
    "grief": [("sadness", 1.0)],
    "joy": [("joy", 1.0)],
    "love": [("love", 1.0)],
    "nervousness": [("fear", 0.8), ("confusion", 0.2)],
    "optimism": [("joy", 0.6), ("excitement", 0.4)],
    "pride": [("joy", 0.7), ("love", 0.3)],
    "realization": [("surprise", 0.6), ("confusion", 0.4)],
    "relief": [("calm", 0.6), ("joy", 0.4)],
    "remorse": [("sadness", 0.7), ("fear", 0.3)],
    "sadness": [("sadness", 1.0)],
    "surprise": [("surprise", 1.0)],
    "neutral": [("neutral", 1.0)],
}

# Inverse mapping: Miyraa → GoEmotions (for data generation)
MIYRAA_TO_GOEMOTIONS = {
    "joy": ["amusement", "joy", "approval", "optimism", "pride", "relief"],
    "love": ["love", "caring", "admiration", "gratitude"],
    "surprise": ["surprise", "realization", "curiosity"],
    "sadness": ["sadness", "grief", "disappointment", "embarrassment", "remorse"],
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness", "embarrassment"],
    "disgust": ["disgust", "disapproval"],
    "calm": ["relief", "approval"],
    "excitement": ["excitement", "desire", "curiosity", "optimism"],
    "confusion": ["confusion", "nervousness", "realization"],
    "neutral": ["neutral"],
}

# =============================================================================
# VAD (VALENCE-AROUSAL-DOMINANCE) DIMENSIONS
# =============================================================================

# VAD values for each core emotion (normalized 0-1)
# Based on established emotion psychology research
EMOTION_VAD_MAPPING = {
    "joy": {"valence": 0.85, "arousal": 0.65, "dominance": 0.70},
    "love": {"valence": 0.90, "arousal": 0.50, "dominance": 0.60},
    "surprise": {"valence": 0.50, "arousal": 0.80, "dominance": 0.40},
    "sadness": {"valence": 0.15, "arousal": 0.30, "dominance": 0.25},
    "anger": {"valence": 0.10, "arousal": 0.85, "dominance": 0.80},
    "fear": {"valence": 0.20, "arousal": 0.75, "dominance": 0.15},
    "disgust": {"valence": 0.15, "arousal": 0.50, "dominance": 0.50},
    "calm": {"valence": 0.70, "arousal": 0.20, "dominance": 0.55},
    "excitement": {"valence": 0.80, "arousal": 0.90, "dominance": 0.65},
    "confusion": {"valence": 0.35, "arousal": 0.55, "dominance": 0.30},
    "neutral": {"valence": 0.50, "arousal": 0.50, "dominance": 0.50},
}

# =============================================================================
# SAFETY CATEGORIES (4 Classes)
# =============================================================================

SAFETY_CATEGORIES = [
    "toxic",         # Harmful, offensive, abusive language
    "profane",       # Profanity, vulgar language
    "threatening",   # Threats of violence or harm
    "harassment",    # Bullying, intimidation, personal attacks
]

# Keywords for safety detection (rule-based fallback)
SAFETY_KEYWORDS = {
    "toxic": ["hate", "idiot", "stupid", "worthless", "kill yourself"],
    "profane": ["fuck", "shit", "damn", "hell", "ass", "bitch"],
    "threatening": ["kill you", "hurt you", "destroy you", "attack"],
    "harassment": ["loser", "ugly", "pathetic", "shut up", "nobody likes"],
}

# =============================================================================
# STYLE CATEGORIES (5 Classes)
# =============================================================================

STYLE_CATEGORIES = [
    "formal",        # Professional, structured, polite
    "casual",        # Relaxed, informal, conversational
    "assertive",     # Direct, confident, commanding
    "empathetic",    # Understanding, supportive, caring
    "humorous",      # Funny, witty, playful
]

# =============================================================================
# INTENT CATEGORIES (6 Classes)
# =============================================================================

INTENT_CATEGORIES = [
    "statement",     # Declarative, informative
    "question",      # Asking for information
    "request",       # Asking for action
    "command",       # Directive, imperative
    "expression",    # Emotional expression
    "social",        # Greeting, thanks, apology
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def map_goemotions_label(go_emotion: str) -> List[Tuple[str, float]]:
    """Convert GoEmotions label to Miyraa emotion(s) with confidence.
    
    Args:
        go_emotion: GoEmotions category name
        
    Returns:
        List of (miyraa_emotion, confidence) tuples
    """
    return GOEMOTIONS_TO_MIYRAA.get(go_emotion, [("neutral", 0.5)])


def get_vad_for_emotion(emotion: str) -> Dict[str, float]:
    """Get VAD values for a given emotion.
    
    Args:
        emotion: Miyraa emotion name
        
    Returns:
        Dictionary with valence, arousal, dominance values (0-1)
    """
    return EMOTION_VAD_MAPPING.get(emotion, {"valence": 0.5, "arousal": 0.5, "dominance": 0.5})


def is_safe_text(text: str, threshold: int = 1) -> bool:
    """Quick rule-based safety check.
    
    Args:
        text: Input text
        threshold: Number of keyword matches to flag as unsafe
        
    Returns:
        True if text appears safe, False otherwise
    """
    text_lower = text.lower()
    total_matches = 0
    
    for category, keywords in SAFETY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                total_matches += 1
                if total_matches >= threshold:
                    return False
    
    return True


def get_emotion_category_mapping() -> Dict[str, List[str]]:
    """Get emotion categories grouped by valence.
    
    Returns:
        Dictionary mapping valence categories to emotion lists
    """
    return {
        "positive": ["joy", "love", "calm", "excitement"],
        "negative": ["sadness", "anger", "fear", "disgust"],
        "ambiguous": ["surprise", "confusion"],
        "neutral": ["neutral"],
    }


def validate_emotion_label(emotion: str) -> bool:
    """Check if emotion label is valid in Miyraa taxonomy.
    
    Args:
        emotion: Emotion label to validate
        
    Returns:
        True if valid, False otherwise
    """
    return emotion in EMOTIONS


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_taxonomy_json(output_path: str):
    """Export taxonomy to JSON for external use.
    
    Args:
        output_path: Path to save JSON file
    """
    taxonomy = {
        "emotions": {
            "labels": EMOTIONS,
            "descriptions": EMOTION_DESCRIPTIONS,
            "vad_mapping": EMOTION_VAD_MAPPING,
        },
        "mappings": {
            "goemotions_to_miyraa": GOEMOTIONS_TO_MIYRAA,
            "miyraa_to_goemotions": MIYRAA_TO_GOEMOTIONS,
        },
        "safety": {
            "categories": SAFETY_CATEGORIES,
            "keywords": SAFETY_KEYWORDS,
        },
        "style": STYLE_CATEGORIES,
        "intent": INTENT_CATEGORIES,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Print taxonomy summary
    print("=== MIYRAA EMOTION TAXONOMY ===\n")
    print(f"Core Emotions ({len(EMOTIONS)}):")
    for emotion in EMOTIONS:
        vad = get_vad_for_emotion(emotion)
        print(f"  - {emotion:12s} | V:{vad['valence']:.2f} A:{vad['arousal']:.2f} D:{vad['dominance']:.2f}")
    
    print(f"\nGoEmotions Mapping: {len(GOEMOTIONS_TO_MIYRAA)} categories")
    print(f"Safety Categories: {len(SAFETY_CATEGORIES)}")
    print(f"Style Categories: {len(STYLE_CATEGORIES)}")
    print(f"Intent Categories: {len(INTENT_CATEGORIES)}")
    
    # Test mappings
    print("\n=== EXAMPLE MAPPINGS ===")
    test_emotions = ["amusement", "disappointment", "nervousness"]
    for ge in test_emotions:
        mapped = map_goemotions_label(ge)
        print(f"GoEmotions '{ge}' → {mapped}")
