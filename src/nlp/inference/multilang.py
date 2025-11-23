"""
Multi-language support with XLM-RoBERTa.
Supports 100+ languages with automatic language detection.
"""

from typing import Optional, List, Dict, Any
import re


# Language detection patterns (simple heuristics)
LANGUAGE_PATTERNS = {
    "en": r"[a-zA-Z\s]+",
    "es": r"[a-záéíóúñü\s]+",
    "fr": r"[a-zàâçéèêëïîôùûüÿæœ\s]+",
    "de": r"[a-zäöüß\s]+",
    "it": r"[a-zàèéìíîòóùú\s]+",
    "pt": r"[a-zàáâãçéêíóôõú\s]+",
    "nl": r"[a-zäëïöü\s]+",
    "ru": r"[а-яё\s]+",
    "zh": r"[\u4e00-\u9fff]+",
    "ja": r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+",
    "ko": r"[\uac00-\ud7af]+",
    "ar": r"[\u0600-\u06ff\s]+",
    "hi": r"[\u0900-\u097f\s]+",
    "th": r"[\u0e00-\u0e7f\s]+",
    "vi": r"[a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]+",
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "unknown": "Unknown",
}


class LanguageDetector:
    """Simple rule-based language detector"""

    def __init__(self):
        self.patterns = {
            lang: re.compile(pattern, re.IGNORECASE)
            for lang, pattern in LANGUAGE_PATTERNS.items()
        }

    def detect(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Detect language of text.

        Args:
            text: Input text
            top_k: Return top-k language candidates

        Returns:
            List of dicts with 'language', 'name', 'confidence'
        """
        if not text or not text.strip():
            return [{"language": "unknown", "name": "Unknown", "confidence": 0.0}]

        # Count matches for each language
        scores = {}
        text_len = len(text)

        for lang, pattern in self.patterns.items():
            matches = pattern.findall(text.lower())
            if matches:
                match_len = sum(len(m) for m in matches)
                scores[lang] = match_len / text_len
            else:
                scores[lang] = 0.0

        # Sort by score
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for lang, score in sorted_langs[:top_k]:
            if score > 0.1:  # Threshold
                results.append(
                    {
                        "language": lang,
                        "name": LANGUAGE_NAMES.get(lang, lang),
                        "confidence": round(score, 3),
                    }
                )

        if not results:
            return [{"language": "unknown", "name": "Unknown", "confidence": 0.0}]

        return results


class MultiLanguagePreprocessor:
    """Language-specific text preprocessing"""

    def __init__(self):
        self.detector = LanguageDetector()

        # Language-specific stopwords (sample)
        self.stopwords = {
            "en": {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "is",
                "are",
                "was",
                "were",
            },
            "es": {
                "el",
                "la",
                "los",
                "las",
                "un",
                "una",
                "y",
                "o",
                "en",
                "de",
                "del",
                "con",
                "por",
                "para",
                "es",
                "son",
                "fue",
                "fueron",
            },
            "fr": {
                "le",
                "la",
                "les",
                "un",
                "une",
                "et",
                "ou",
                "dans",
                "de",
                "du",
                "avec",
                "par",
                "pour",
                "est",
                "sont",
                "était",
                "étaient",
            },
            "de": {
                "der",
                "die",
                "das",
                "ein",
                "eine",
                "und",
                "oder",
                "in",
                "von",
                "mit",
                "zu",
                "für",
                "ist",
                "sind",
                "war",
                "waren",
            },
        }

    def preprocess(
        self, text: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Preprocess text with language-specific rules.

        Args:
            text: Input text
            language: Language code (auto-detect if None)

        Returns:
            Dict with 'text', 'language', 'original_text'
        """
        original_text = text

        # Detect language if not provided
        if language is None:
            detected = self.detector.detect(text, top_k=1)
            language = detected[0]["language"]

        # Basic normalization (universal)
        text = text.strip()

        # Language-specific preprocessing
        if language in ["zh", "ja", "ko"]:
            # Asian languages - no lowercasing
            pass
        elif language in ["ar", "he"]:
            # RTL languages - preserve direction
            pass
        else:
            # Latin-based languages - lowercase
            text = text.lower()

        return {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
            "original_text": original_text,
        }

    def get_stopwords(self, language: str) -> set:
        """Get stopwords for language"""
        return self.stopwords.get(language, set())


class MultiLanguageInference:
    """
    Multi-language inference wrapper.
    Uses XLM-RoBERTa for cross-lingual understanding.
    """

    def __init__(self, model_name: str = "xlm-roberta-base"):
        """
        Initialize multi-language inference.

        Args:
            model_name: HuggingFace model name (default: xlm-roberta-base)
        """
        self.model_name = model_name
        self.preprocessor = MultiLanguagePreprocessor()

        # Lazy load model (not implemented here, just placeholder)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load XLM-RoBERTa model (placeholder)"""
        # In production, load actual model:
        # from transformers import AutoModel, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name)
        print(f"Model {self.model_name} loaded (placeholder)")

    def predict(
        self,
        text: str,
        language: Optional[str] = None,
        task: str = "emotion",
    ) -> Dict[str, Any]:
        """
        Predict emotion/sentiment in any language.

        Args:
            text: Input text
            language: Language code (auto-detect if None)
            task: Task type ('emotion' or 'sentiment')

        Returns:
            Prediction results with language info
        """
        # Preprocess
        processed = self.preprocessor.preprocess(text, language)

        # In production, use actual model:
        # inputs = self.tokenizer(processed['text'], return_tensors='pt')
        # outputs = self.model(**inputs)
        # predictions = classify(outputs)

        # Placeholder result
        result = {
            "text": processed["text"],
            "language": processed["language"],
            "language_name": processed["language_name"],
            "predictions": [
                {"label": "joy", "score": 0.85},
                {"label": "neutral", "score": 0.10},
            ],
            "task": task,
        }

        return result

    def predict_batch(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None,
        task: str = "emotion",
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple texts.

        Args:
            texts: List of input texts
            languages: List of language codes (auto-detect if None)
            task: Task type

        Returns:
            List of prediction results
        """
        if languages is None:
            languages = [None] * len(texts)

        results = []
        for text, lang in zip(texts, languages):
            result = self.predict(text, language=lang, task=task)
            results.append(result)

        return results


class LanguageRouter:
    """
    Routes requests to language-specific models or endpoints.
    Useful for handling language-specific fine-tuned models.
    """

    def __init__(self):
        self.detector = LanguageDetector()
        self.language_models = {}  # lang -> model mapping

    def register_model(self, language: str, model: Any) -> None:
        """Register language-specific model"""
        self.language_models[language] = model
        print(f"Registered model for {LANGUAGE_NAMES.get(language, language)}")

    def route(self, text: str, task: str = "emotion") -> Dict[str, Any]:
        """
        Route to appropriate model based on language.

        Args:
            text: Input text
            task: Task type

        Returns:
            Prediction results
        """
        # Detect language
        detected = self.detector.detect(text, top_k=1)
        language = detected[0]["language"]

        # Get model for language (or use default)
        model = self.language_models.get(language)

        if model is None:
            # Use multilingual model (XLM-R)
            model = self.language_models.get("multilingual")

        if model is None:
            return {
                "error": "No model available",
                "language": language,
                "language_name": LANGUAGE_NAMES.get(language, language),
            }

        # Make prediction (placeholder)
        result = {
            "text": text,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
            "predictions": [{"label": "joy", "score": 0.85}],
            "task": task,
        }

        return result


if __name__ == "__main__":
    # Demo
    print("=== Language Detection ===")
    detector = LanguageDetector()

    test_texts = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hallo, wie geht es dir heute?",
        "你好，你今天好吗？",
        "こんにちは、今日はどうですか？",
        "안녕하세요, 오늘 어떻게 지내세요?",
        "مرحبا، كيف حالك اليوم؟",
    ]

    for text in test_texts:
        results = detector.detect(text, top_k=1)
        lang = results[0]
        print(f"{text[:30]:<30} -> {lang['name']} ({lang['confidence']:.2f})")

    print("\n=== Multi-Language Preprocessing ===")
    preprocessor = MultiLanguagePreprocessor()

    for text in test_texts[:4]:
        result = preprocessor.preprocess(text)
        print(
            f"{result['language_name']:<10} | {result['text'][:40]}"
        )

    print("\n=== Multi-Language Inference ===")
    inference = MultiLanguageInference()
    inference.load_model()

    result = inference.predict("I am so happy today!", language="en")
    print(f"\nInput: {result['text']}")
    print(f"Language: {result['language_name']}")
    print(f"Predictions: {result['predictions']}")

    # Batch
    print("\n=== Batch Prediction ===")
    batch_texts = [
        "I love this!",
        "Me encanta esto!",
        "J'adore ça!",
    ]
    batch_results = inference.predict_batch(batch_texts)

    for result in batch_results:
        print(f"{result['language_name']:<10} | {result['text']}")
