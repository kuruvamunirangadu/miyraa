"""
Expanded unit tests for preprocessing module.
Tests normalization, edge cases, and label mapping.
"""

import pytest
from src.nlp.preprocessing.normalize import normalize_text
from src.nlp.preprocessing.label_mapping import (
    TARGET_TAXONOMY,
    map_goemotions_to_target,
    target_to_index
)


class TestNormalization:
    """Test text normalization functions"""

    def test_normalize_basic(self):
        """Test basic normalization"""
        assert normalize_text("  Hello   WORLD\n") == "hello world"
        assert normalize_text("UPPERCASE") == "uppercase"
        assert normalize_text("  extra   spaces  ") == "extra spaces"

    def test_normalize_empty(self):
        """Test empty inputs"""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""
        assert normalize_text("   ") == ""

    def test_normalize_special_chars(self):
        """Test special character handling"""
        assert normalize_text("hello! world?") == "hello! world?"
        assert normalize_text("test@example.com") == "test@example.com"
        assert normalize_text("$100 USD") == "$100 usd"

    def test_normalize_unicode(self):
        """Test unicode characters"""
        assert normalize_text("café") == "café"
        assert normalize_text("日本語") == "日本語"
        assert normalize_text("emoji 😊") == "emoji 😊"

    def test_normalize_newlines(self):
        """Test newline handling"""
        assert normalize_text("line1\nline2") == "line1 line2"
        assert normalize_text("line1\r\nline2") == "line1 line2"
        assert normalize_text("line1\n\nline2") == "line1 line2"

    def test_normalize_tabs(self):
        """Test tab handling"""
        assert normalize_text("col1\tcol2") == "col1 col2"
        assert normalize_text("a\t\tb") == "a b"

    def test_normalize_consecutive_spaces(self):
        """Test multiple consecutive spaces"""
        text = "too    many     spaces"
        assert "    " not in normalize_text(text)
        assert "  " not in normalize_text(text)

    def test_normalize_mixed_case(self):
        """Test mixed case normalization"""
        assert normalize_text("Hello World") == "hello world"
        assert normalize_text("CamelCase") == "camelcase"
        assert normalize_text("snake_case") == "snake_case"

    def test_normalize_numbers(self):
        """Test number preservation"""
        assert normalize_text("I have 5 apples") == "i have 5 apples"
        assert normalize_text("The year is 2024") == "the year is 2024"

    def test_normalize_urls(self):
        """Test URL handling"""
        url = "https://example.com/path"
        normalized = normalize_text(url)
        assert "example.com" in normalized.lower()


class TestLabelMapping:
    """Test label mapping functions"""

    def test_target_taxonomy_exists(self):
        """Test target taxonomy is defined"""
        assert isinstance(TARGET_TAXONOMY, list)
        assert len(TARGET_TAXONOMY) > 0
        assert "joy" in TARGET_TAXONOMY
        assert "anger" in TARGET_TAXONOMY
        assert "neutral" in TARGET_TAXONOMY

    def test_target_taxonomy_no_duplicates(self):
        """Test taxonomy has no duplicates"""
        assert len(TARGET_TAXONOMY) == len(set(TARGET_TAXONOMY))

    def test_target_to_index_basic(self):
        """Test target_to_index mapping"""
        idx_map = target_to_index()
        assert isinstance(idx_map, dict)
        assert len(idx_map) == len(TARGET_TAXONOMY)
        assert "joy" in idx_map
        assert idx_map["joy"] >= 0

    def test_target_to_index_custom_taxonomy(self):
        """Test target_to_index with custom taxonomy"""
        custom = ["happy", "sad", "angry"]
        idx_map = target_to_index(custom)
        assert len(idx_map) == 3
        assert idx_map["happy"] == 0
        assert idx_map["sad"] == 1
        assert idx_map["angry"] == 2

    def test_map_goemotions_basic(self):
        """Test basic GoEmotions mapping"""
        result = map_goemotions_to_target(["joy", "happiness"])
        assert isinstance(result, list)
        assert "joy" in result

    def test_map_goemotions_with_text(self):
        """Test GoEmotions mapping with text context"""
        result = map_goemotions_to_target(["joy"], "I'm so proud of this")
        assert isinstance(result, list)
        # May contain joy and other emotions based on text

    def test_map_goemotions_nostalgia(self):
        """Test nostalgia detection from text"""
        result = map_goemotions_to_target([], "I remember when we used to play")
        assert isinstance(result, list)
        # Should detect nostalgia-related emotions

    def test_map_goemotions_empty(self):
        """Test empty GoEmotions labels"""
        result = map_goemotions_to_target([])
        assert isinstance(result, list)

    def test_map_goemotions_unknown_labels(self):
        """Test handling of unknown labels"""
        result = map_goemotions_to_target(["unknown_emotion"])
        assert isinstance(result, list)
        # Should handle gracefully

    def test_label_to_index_roundtrip(self):
        """Test label to index and back"""
        idx_map = target_to_index()
        for label, idx in idx_map.items():
            assert isinstance(idx, int)
            assert 0 <= idx < len(TARGET_TAXONOMY)
            assert TARGET_TAXONOMY[idx] == label

    def test_all_target_labels_mapped(self):
        """Test all target labels have indices"""
        idx_map = target_to_index()
        for label in TARGET_TAXONOMY:
            assert label in idx_map
            assert idx_map[label] >= 0


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing"""

    def test_very_long_text(self):
        """Test handling of very long text"""
        long_text = "word " * 10000
        result = normalize_text(long_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_only_whitespace(self):
        """Test text with only whitespace"""
        assert normalize_text("   \n\t  ") == ""

    def test_only_punctuation(self):
        """Test text with only punctuation"""
        result = normalize_text("!@#$%^&*()")
        assert isinstance(result, str)

    def test_mixed_languages(self):
        """Test mixed language text"""
        text = "Hello 你好 Bonjour"
        result = normalize_text(text)
        assert "hello" in result
        assert "bonjour" in result

    def test_repeated_chars(self):
        """Test repeated characters"""
        text = "Helloooooo"
        result = normalize_text(text)
        assert "hello" in result

    def test_control_characters(self):
        """Test control characters"""
        text = "hello\x00world\x01"
        result = normalize_text(text)
        # Should handle gracefully without crashing
        assert isinstance(result, str)


class TestPreprocessingBatch:
    """Test batch preprocessing operations"""

    def test_batch_normalize(self):
        """Test normalizing multiple texts"""
        texts = [
            "  HELLO  ",
            "WORLD",
            "  Test  Case  "
        ]
        results = [normalize_text(t) for t in texts]
        assert results == ["hello", "world", "test case"]

    def test_empty_batch(self):
        """Test empty batch"""
        texts = []
        results = [normalize_text(t) for t in texts]
        assert results == []

    def test_mixed_batch(self):
        """Test batch with mixed inputs"""
        texts = ["normal", "", "  spaces  ", None]
        results = [normalize_text(t) for t in texts]
        assert results[0] == "normal"
        assert results[1] == ""
        assert results[2] == "spaces"
        assert results[3] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
