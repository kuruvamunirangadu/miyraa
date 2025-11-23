"""
Text summarization module (optional head).
Provides extractive summarization for long texts.
"""

from typing import List, Dict, Optional, Tuple
import re
import math


class TextSummarizer:
    """Extractive text summarization using sentence scoring"""

    def __init__(self, max_sentences: int = 3, min_sentence_length: int = 10):
        """
        Initialize summarizer.

        Args:
            max_sentences: Maximum number of sentences in summary
            min_sentence_length: Minimum characters for a valid sentence
        """
        self.max_sentences = max_sentences
        self.min_sentence_length = min_sentence_length

        # Common stop words (simplified list)
        self.stop_words = set([
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
        ])

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        return_scores: bool = False,
    ) -> Dict:
        """
        Generate extractive summary.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (default: max_sentences)
            return_scores: Whether to return sentence scores

        Returns:
            Dictionary with summary and metadata
        """
        if not text or len(text) < self.min_sentence_length:
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "compression_ratio": 1.0,
                "num_sentences": 0,
            }

        num_sentences = num_sentences or self.max_sentences

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "compression_ratio": 1.0,
                "num_sentences": len(sentences),
            }

        # Score sentences
        scores = self._score_sentences(sentences)

        # Select top sentences
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:num_sentences]

        # Maintain original order
        top_indices.sort()
        summary_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(summary_sentences)

        result = {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text),
            "num_sentences": len(summary_sentences),
            "selected_indices": top_indices,
        }

        if return_scores:
            result["scores"] = scores

        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (can be improved with nltk/spacy)
        sentences = re.split(r"[.!?]+\s+", text)

        # Filter out short sentences
        sentences = [
            s.strip()
            for s in sentences
            if len(s.strip()) >= self.min_sentence_length
        ]

        return sentences

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences based on multiple factors"""
        if not sentences:
            return []

        # Calculate word frequencies
        word_freq = self._calculate_word_frequencies(sentences)

        scores = []
        for sentence in sentences:
            score = self._score_sentence(sentence, word_freq, len(sentences))
            scores.append(score)

        # Normalize scores
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]

        return scores

    def _score_sentence(
        self, sentence: str, word_freq: Dict[str, float], total_sentences: int
    ) -> float:
        """Score a single sentence"""
        words = self._tokenize(sentence.lower())

        if not words:
            return 0.0

        # 1. Word frequency score (importance of words)
        freq_score = sum(word_freq.get(word, 0.0) for word in words) / len(words)

        # 2. Sentence position score (first sentences are often important)
        # This is handled during scoring, not here

        # 3. Sentence length score (prefer medium-length sentences)
        length_score = self._length_score(len(words))

        # 4. Keyword density (ratio of non-stop words)
        non_stop_words = [w for w in words if w not in self.stop_words]
        keyword_density = len(non_stop_words) / len(words) if words else 0

        # Combine scores
        total_score = (
            freq_score * 0.5 + length_score * 0.2 + keyword_density * 0.3
        )

        return total_score

    def _calculate_word_frequencies(
        self, sentences: List[str]
    ) -> Dict[str, float]:
        """Calculate word frequencies across all sentences"""
        word_count = {}
        total_words = 0

        for sentence in sentences:
            words = self._tokenize(sentence.lower())
            for word in words:
                if word not in self.stop_words:
                    word_count[word] = word_count.get(word, 0) + 1
                    total_words += 1

        # Normalize to frequencies
        word_freq = {
            word: count / total_words if total_words > 0 else 0.0
            for word, count in word_count.items()
        }

        return word_freq

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        # Remove punctuation and split
        words = re.findall(r"\b\w+\b", text)
        return [w for w in words if len(w) > 1]

    def _length_score(self, num_words: int) -> float:
        """Score based on sentence length (prefer 10-25 words)"""
        optimal_length = 17
        if num_words < 5:
            return 0.3
        elif num_words > 40:
            return 0.5

        # Gaussian-like score around optimal length
        diff = abs(num_words - optimal_length)
        score = math.exp(-((diff / 10) ** 2))
        return score

    def batch_summarize(
        self, texts: List[str], **kwargs
    ) -> List[Dict]:
        """Summarize multiple texts"""
        return [self.summarize(text, **kwargs) for text in texts]

    def summarize_by_ratio(
        self, text: str, ratio: float = 0.3
    ) -> Dict:
        """
        Summarize to a target compression ratio.

        Args:
            text: Input text
            ratio: Target compression ratio (0.0-1.0)

        Returns:
            Summary dictionary
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return self.summarize(text)

        num_sentences = max(1, int(len(sentences) * ratio))
        return self.summarize(text, num_sentences=num_sentences)


class BulletPointSummarizer(TextSummarizer):
    """Generate bullet-point style summaries"""

    def summarize_to_bullets(
        self, text: str, max_bullets: int = 5
    ) -> Dict:
        """
        Generate bullet-point summary.

        Args:
            text: Input text
            max_bullets: Maximum number of bullet points

        Returns:
            Dictionary with bullet points
        """
        result = self.summarize(text, num_sentences=max_bullets)

        # Convert sentences to bullet points
        sentences = self._split_sentences(result["summary"])
        bullets = [f"• {s}" for s in sentences]

        return {
            **result,
            "bullets": bullets,
            "num_bullets": len(bullets),
        }


def get_summarizer(
    max_sentences: int = 3, min_sentence_length: int = 10
) -> TextSummarizer:
    """Factory function to get summarizer instance"""
    return TextSummarizer(
        max_sentences=max_sentences, min_sentence_length=min_sentence_length
    )


def get_bullet_summarizer(max_bullets: int = 5) -> BulletPointSummarizer:
    """Factory function to get bullet-point summarizer"""
    return BulletPointSummarizer(
        max_sentences=max_bullets, min_sentence_length=10
    )


if __name__ == "__main__":
    # Demo
    summarizer = get_summarizer(max_sentences=2)

    test_text = """
    Artificial intelligence is transforming the way we live and work.
    Machine learning algorithms can now recognize patterns in vast amounts of data.
    Natural language processing enables computers to understand human language.
    Deep learning has achieved remarkable results in image and speech recognition.
    The ethical implications of AI are becoming increasingly important to consider.
    Researchers are working on making AI systems more transparent and explainable.
    """

    result = summarizer.summarize(test_text.strip())
    print("Original text length:", result["original_length"])
    print("Summary length:", result["summary_length"])
    print("Compression ratio:", f"{result['compression_ratio']:.2%}")
    print("\nSummary:")
    print(result["summary"])

    # Bullet points
    bullet_summarizer = get_bullet_summarizer(max_bullets=3)
    bullet_result = bullet_summarizer.summarize_to_bullets(test_text.strip())
    print("\nBullet Points:")
    for bullet in bullet_result["bullets"]:
        print(bullet)
