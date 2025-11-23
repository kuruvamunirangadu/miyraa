"""
Emotion explanation module with keyword-level attention.
Provides interpretability by highlighting which words/phrases contributed to emotion detection.
"""

from typing import List, Dict, Tuple, Optional
import re
import numpy as np


class EmotionExplainer:
    """Generates explanations for emotion predictions using attention-like scoring"""

    # Emotion-specific keyword patterns
    EMOTION_KEYWORDS = {
        "joy": [
            r"\b(happy|joy|glad|delight|pleasure|cheer|excit|thrilled|wonderful|amazing|love)\w*\b",
            r"\b(great|good|best|awesome|fantastic|excellent|brilliant)\b",
            r"[!]{1,3}(?!\w)",  # Exclamation marks
            r"😊|😄|😃|🎉|❤️|💖|🥰",  # Happy emojis
        ],
        "sadness": [
            r"\b(sad|unhappy|depress|miserable|sorrow|grief|disappoint|upset)\w*\b",
            r"\b(cry|tear|hurt|pain|lonely|alone|miss)\w*\b",
            r"😢|😭|💔|☹️|😞",
        ],
        "anger": [
            r"\b(angry|anger|furious|mad|rage|annoyed|irritat|frustrat)\w*\b",
            r"\b(hate|damn|stupid|idiot|terrible|awful)\w*\b",
            r"😠|😡|🤬|💢",
        ],
        "fear": [
            r"\b(fear|afraid|scared|terrif|anxious|worry|nervous|panic)\w*\b",
            r"\b(danger|threat|risk|concern)\w*\b",
            r"😨|😰|😱",
        ],
        "surprise": [
            r"\b(surprise|amaz|shock|astonish|unexpected|sudden)\w*\b",
            r"\b(wow|omg|whoa)\b",
            r"😮|😲|🤯",
        ],
        "disgust": [
            r"\b(disgust|gross|nasty|sick|revolting|repulsive)\w*\b",
            r"\b(yuck|eww|ugh)\b",
            r"🤢|🤮|😖",
        ],
        "love": [
            r"\b(love|adore|cherish|affection|fond|care)\w*\b",
            r"\b(heart|dear|darling|sweetheart)\w*\b",
            r"❤️|💕|💖|💗|💘|💝|😍|🥰",
        ],
        "neutral": [
            r"\b(okay|ok|fine|alright|normal|regular)\b",
            r"\b(think|believe|consider|seem)\w*\b",
        ],
    }

    # Intensity modifiers
    INTENSIFIERS = [
        r"\b(very|really|extremely|incredibly|absolutely|completely|totally|so)\b",
        r"\b(quite|rather|pretty|fairly)\b",
    ]

    NEGATIONS = [
        r"\b(not|no|never|nothing|none|neither|nor|nobody|nowhere)\b",
        r"\b(don't|doesn't|didn't|won't|wouldn't|can't|cannot|shouldn't)\b",
    ]

    def __init__(self):
        """Initialize the explainer"""
        self.compiled_patterns = {}
        for emotion, patterns in self.EMOTION_KEYWORDS.items():
            self.compiled_patterns[emotion] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        self.intensifier_pattern = re.compile(
            "|".join(self.INTENSIFIERS), re.IGNORECASE
        )
        self.negation_pattern = re.compile("|".join(self.NEGATIONS), re.IGNORECASE)

    def explain(
        self,
        text: str,
        predicted_emotion: str,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> Dict:
        """
        Generate explanation for emotion prediction.

        Args:
            text: Input text
            predicted_emotion: Predicted emotion label
            top_k: Number of top keywords to return
            min_score: Minimum score threshold for keywords

        Returns:
            Dictionary with explanation details:
            {
                "keywords": [(word, score, span), ...],
                "emotion": predicted_emotion,
                "confidence": overall_confidence,
                "has_negation": bool,
                "has_intensifier": bool
            }
        """
        if not text:
            return {
                "keywords": [],
                "emotion": predicted_emotion,
                "confidence": 0.0,
                "has_negation": False,
                "has_intensifier": False,
            }

        # Find keyword matches
        keyword_scores = self._score_keywords(text, predicted_emotion)

        # Detect modifiers
        has_negation = bool(self.negation_pattern.search(text))
        has_intensifier = bool(self.intensifier_pattern.search(text))

        # Sort by score and take top-k
        sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
        top_keywords = [kw for kw in sorted_keywords if kw[1] >= min_score][:top_k]

        # Calculate overall confidence
        confidence = self._calculate_confidence(top_keywords, has_intensifier)

        return {
            "keywords": top_keywords,
            "emotion": predicted_emotion,
            "confidence": confidence,
            "has_negation": has_negation,
            "has_intensifier": has_intensifier,
            "explanation": self._generate_text_explanation(
                top_keywords, predicted_emotion, has_negation, has_intensifier
            ),
        }

    def _score_keywords(
        self, text: str, emotion: str
    ) -> List[Tuple[str, float, Tuple[int, int]]]:
        """
        Score keywords in text based on emotion patterns.

        Returns:
            List of (keyword, score, (start, end)) tuples
        """
        keywords = []

        if emotion not in self.compiled_patterns:
            return keywords

        patterns = self.compiled_patterns[emotion]

        for pattern in patterns:
            for match in pattern.finditer(text):
                keyword = match.group()
                span = match.span()

                # Base score
                score = 0.5

                # Adjust score based on keyword characteristics
                if len(keyword) > 3:
                    score += 0.1

                # Check for nearby intensifiers
                context_start = max(0, span[0] - 20)
                context_end = min(len(text), span[1] + 20)
                context = text[context_start:context_end]

                if self.intensifier_pattern.search(context):
                    score += 0.3

                # Check for negations (reduces score)
                if self.negation_pattern.search(context):
                    score *= 0.3

                # Emoji bonus
                if any(emoji in keyword for emoji in ["😊", "😢", "😠", "❤️"]):
                    score += 0.2

                keywords.append((keyword, min(score, 1.0), span))

        return keywords

    def _calculate_confidence(
        self, keywords: List[Tuple], has_intensifier: bool
    ) -> float:
        """Calculate overall confidence from keyword scores"""
        if not keywords:
            return 0.0

        # Average of top keyword scores
        avg_score = np.mean([kw[1] for kw in keywords])

        # Number of keywords found
        num_keywords = len(keywords)

        # Combine factors
        confidence = avg_score * 0.7 + min(num_keywords / 5.0, 1.0) * 0.3

        # Intensifier boost
        if has_intensifier:
            confidence = min(confidence * 1.2, 1.0)

        return float(confidence)

    def _generate_text_explanation(
        self,
        keywords: List[Tuple],
        emotion: str,
        has_negation: bool,
        has_intensifier: bool,
    ) -> str:
        """Generate human-readable explanation"""
        if not keywords:
            return f"Predicted '{emotion}' with low confidence (no clear indicators)"

        keyword_list = [kw[0] for kw in keywords[:3]]
        keyword_str = ", ".join(f"'{kw}'" for kw in keyword_list)

        explanation = f"Predicted '{emotion}' based on keywords: {keyword_str}"

        if has_intensifier:
            explanation += " (with intensifiers)"
        if has_negation:
            explanation += " (negation detected - may reverse sentiment)"

        return explanation

    def highlight_text(
        self, text: str, keywords: List[Tuple[str, float, Tuple[int, int]]]
    ) -> str:
        """
        Generate HTML with highlighted keywords.

        Args:
            text: Original text
            keywords: List of (keyword, score, (start, end)) tuples

        Returns:
            HTML string with <mark> tags
        """
        if not keywords:
            return text

        # Sort by position
        sorted_keywords = sorted(keywords, key=lambda x: x[2][0])

        html_parts = []
        last_end = 0

        for keyword, score, (start, end) in sorted_keywords:
            # Add text before keyword
            html_parts.append(text[last_end:start])

            # Add highlighted keyword with score-based color intensity
            opacity = int(score * 100)
            color = f"rgba(255, 235, 59, {opacity / 100})"  # Yellow with opacity
            html_parts.append(
                f'<mark style="background-color: {color};" '
                f'title="Score: {score:.2f}">{text[start:end]}</mark>'
            )

            last_end = end

        # Add remaining text
        html_parts.append(text[last_end:])

        return "".join(html_parts)

    def batch_explain(
        self, texts: List[str], emotions: List[str], **kwargs
    ) -> List[Dict]:
        """Explain multiple texts at once"""
        return [
            self.explain(text, emotion, **kwargs)
            for text, emotion in zip(texts, emotions)
        ]


def get_explainer() -> EmotionExplainer:
    """Factory function to get explainer instance"""
    return EmotionExplainer()


if __name__ == "__main__":
    # Demo
    explainer = EmotionExplainer()

    test_cases = [
        ("I'm so happy and excited about this!", "joy"),
        ("This is extremely disappointing and sad", "sadness"),
        ("I'm not happy at all", "sadness"),
        ("What an absolutely terrible experience!", "anger"),
        ("I love this so much! ❤️", "love"),
    ]

    for text, emotion in test_cases:
        result = explainer.explain(text, emotion)
        print(f"\nText: {text}")
        print(f"Emotion: {emotion}")
        print(f"Keywords: {result['keywords'][:3]}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        print(f"Highlighted: {explainer.highlight_text(text, result['keywords'])}")
