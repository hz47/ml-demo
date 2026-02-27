import pytest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.word_analysis import get_top_words, run_word_analysis


class TestWordAnalysis:

    def test_get_top_words_returns_list(self):
        texts = ["hello world hello", "world is great"]
        result = get_top_words(texts, 10)
        assert isinstance(result, list)

    def test_get_top_words_counts_correctly(self):
        texts = ["hello world hello", "hello world"]
        result = get_top_words(texts, 10)
        word_dict = dict(result)
        assert word_dict.get("hello", 0) == 3
        assert word_dict.get("world", 0) == 2

    def test_get_top_words_respects_n(self):
        texts = ["one two three four five six seven eight nine ten"]
        result = get_top_words(texts, 3)
        assert len(result) == 3

    def test_get_top_words_empty_texts(self):
        result = get_top_words([], 10)
        assert result == []

    def test_get_top_words_case_insensitive(self):
        texts = ["Hello World", "HELLO world"]
        result = get_top_words(texts, 10)
        word_dict = dict(result)
        assert "hello" in word_dict
        assert "HELLO" not in word_dict

    def test_get_top_words_empty_string(self):
        texts = [""]
        result = get_top_words(texts, 10)
        assert result == []

    def test_get_top_words_single_word(self):
        texts = ["hello"]
        result = get_top_words(texts, 5)
        assert len(result) == 1
        assert result[0] == ("hello", 1)


class TestRunWordAnalysis:

    @patch("utils.word_analysis.pd.read_csv")
    def test_run_word_analysis(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "clean_strict": ["hello world hello", "world is great", "hello test"]
        })
        
        result = run_word_analysis(top_n=5)
        
        assert isinstance(result, list)
        assert len(result) <= 5
