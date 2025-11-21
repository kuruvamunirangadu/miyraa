"""Label mapping utilities: GoEmotions -> target taxonomy with rule-based nostalgia/pride.

This module defines a target taxonomy and a mapping helper. It intentionally keeps
the mapping configurable: if your project has a different target set, update
`TARGET_TAXONOMY` and `GOEMOTIONS_TO_TARGET`.

Functions:
  - map_goemotions_to_target(go_labels: list[str], text: str|None) -> list[str]
    returns list of target label names.
  - target_to_index(taxonomy) -> dict name->idx helper

Rule-based additions:
  - "pride": flagged if source labels include 'pride' or text contains '\bproud\b' or 'pride'
  - "nostalgia": flagged if text contains keywords like 'nostalg', 'remember when', 'used to', 'back in the day'
"""
from typing import List, Dict, Optional
import re

# Target taxonomy (example). This mirrors a typical emotion taxonomy but adds
# 'nostalgia' and keeps 'pride' (if present in source); edit as needed.
# A compact target taxonomy (11 emotions) intended for the project's multi-head
# classifier. The set below was chosen to be intuitive and compact for mobile
# and low-latency deployments. Edit if you prefer a different set.
TARGET_TAXONOMY: List[str] = [
    "joy",
    "love",
    "calm",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "anticipation",
    "trust",
    "neutral",
]


# Mapping from common GoEmotions labels to our compact taxonomy. This is a
# best-effort mapping: when a GoEmotions label clearly corresponds to one of
# our target labels we map it; otherwise it can be ignored or mapped to the
# closest label. Extend this dict if your source labels differ.
GOEMOTIONS_TO_TARGET: Dict[str, str] = {
    # direct semantic matches
    "joy": "joy",
    "happiness": "joy",
    "amusement": "joy",
    "love": "love",
    "affection": "love",
    "caring": "love",
    "calm": "calm",
    "relief": "calm",
    "contentment": "calm",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "frustration": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "surprise": "surprise",
    "excitement": "surprise",
    "disgust": "disgust",
    "revulsion": "disgust",
    "anticipation": "anticipation",
    "optimism": "anticipation",
    "trust": "trust",
    "approval": "trust",
    # default fallback: map unknown/neutral-ish labels to neutral
    "neutral": "neutral",
}


def target_to_index(taxonomy: Optional[List[str]] = None) -> Dict[str, int]:
    taxonomy = taxonomy or TARGET_TAXONOMY
    return {n: i for i, n in enumerate(taxonomy)}


_NOSTALGIA_PAT = re.compile(r"\b(nostalg|remember when|used to|back in the day|take me back|yaad|yaad aa|yaadien)\b", re.I)
# include simple Hinglish/Hindi phrases that commonly indicate pride or nostalgia
_PRIDE_PAT = re.compile(r"\b(proud|pride|so proud|garv|garv se|bahut proud)\b", re.I)

# small hindi/hinglish lexical hints for emotion mapping (examples)
_HINDI_EXAMPLES = {
    "joy": ["bahut khush", "khushi", "bahut accha"],
    "nostalgia": ["yaad aa raha", "yaadien", "purane din"],
    "pride": ["garv"],
}


def map_goemotions_to_target(go_labels: List[str], text: Optional[str] = None) -> List[str]:
    """Map a list of GoEmotions label names to the target taxonomy.

    - go_labels: list of source label names (strings). If empty, mapping may rely on text rules.
    - text: optional text to apply rule-based additions like 'nostalgia' and 'pride'.

    Returns list of unique target label names.
    """
    out = set()
    text = text or ""
    # direct mapping
    for src in (go_labels or []):
        key = src.strip().lower()
        if key in GOEMOTIONS_TO_TARGET:
            out.add(GOEMOTIONS_TO_TARGET[key])

    # rule-based pride detection
    if _PRIDE_PAT.search(text):
        # map pride -> trust/joy depending on tone; choose trust by default
        out.add("trust")

    # rule-based nostalgia detection maps to 'neutral' or 'joy' depending on
    # keywords; for simplicity add to 'neutral' if present (caller can post-map)
    if _NOSTALGIA_PAT.search(text):
        out.add("neutral")

    # small heuristic: check for Hindi/Hinglish example phrases and map
    if text:
        t = text.lower()
        for key, examples in _HINDI_EXAMPLES.items():
            for ex in examples:
                if ex in t:
                    if key == "nostalgia":
                        out.add("neutral")
                    elif key == "pride":
                        out.add("trust")
                    else:
                        out.add(key)

    # if nothing mapped, return ['neutral'] to ensure an explicit label
    if not out:
        return ["neutral"]
    return sorted(out)


if __name__ == "__main__":
    # quick smoke tests
    print(map_goemotions_to_target(["joy", "surprise"], "That was amazing!"))
    print(map_goemotions_to_target([], "I remember when we used to play outside."))
    print(map_goemotions_to_target(["pride"], ""))
