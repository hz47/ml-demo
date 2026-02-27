import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.clean import clean_text_strict, clean_text_light


class TestCleanText:
    def test_clean_text_strict_removes_symbols(self):
        result = clean_text_strict("Hello! How are you?")
        assert "!" not in result
        assert "?" not in result

    def test_clean_text_strict_lowercases(self):
        result = clean_text_strict("HELLO World")
        assert result == "hello world"

    def test_clean_text_strict_removes_stopwords(self):
        result = clean_text_strict("the quick brown fox")
        assert "the" not in result

    def test_clean_text_light_preserves_spam_symbols(self):
        result = clean_text_light("Win $1000! Call now?")
        assert "$" in result
        assert "!" in result

    def test_clean_text_light_handles_empty_string(self):
        result = clean_text_light("")
        assert result == ""

    def test_clean_text_strict_handles_non_string(self):
        result = clean_text_strict(123)
        assert result == ""
