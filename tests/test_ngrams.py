import pytest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ngrams import top_ngrams, run_ngrams


class TestNgrams:

    def test_top_ngrams_returns_list(self):
        texts = [
            "hello world hello world",
            "world is great",
            "hello is great world"
        ]
        result = top_ngrams(texts, (2, 2), 5, min_df=1)
        assert isinstance(result, list)

    def test_top_ngrams_unigrams(self):
        texts = ["hello world hello", "world is great"]
        result = top_ngrams(texts, (1, 1), 5, min_df=1)
        assert "hello" in result or "world" in result

    def test_top_ngrams_empty_texts(self):
        result = top_ngrams([], (1, 1), 5)
        assert result == []

    def test_top_ngrams_single_text(self):
        texts = ["hello world hello"]
        result = top_ngrams(texts, (2, 2), 5, min_df=1)
        assert isinstance(result, list)

    def test_top_ngrams_respects_top_n(self):
        texts = [
            "hello world foo bar baz",
            "hello world foo bar",
            "hello world foo",
            "hello world",
            "hello"
        ]
        result = top_ngrams(texts, (1, 1), 3, min_df=1)
        assert len(result) <= 3

    @pytest.mark.parametrize("ngram_range", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3)])
    def test_top_ngrams_different_ranges(self, ngram_range):
        texts = ["hello world hello world", "world is great"]
        result = top_ngrams(texts, ngram_range, 5, min_df=1)
        assert isinstance(result, list)


class TestRunNgrams:

    @patch("utils.ngrams.pd.read_csv")
    def test_run_ngrams(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "clean_strict": ["hello world hello", "world is great", "hello test"]
        })
        
        bigrams, trigrams = run_ngrams(top_n=5)
        
        assert isinstance(bigrams, list)
        assert isinstance(trigrams, list)
