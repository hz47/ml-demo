import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.clean import clean_text_strict, clean_text_light, get_stopwords, load_raw_data, process_dataframe, run_preprocessing


class TestCleanText:
    @pytest.fixture(autouse=True)
    def setup_stopwords(self):
        self.stop_words = get_stopwords()

    def test_clean_text_strict_removes_symbols(self):
        result = clean_text_strict("Hello! How are you?", self.stop_words)
        assert "!" not in result
        assert "?" not in result

    def test_clean_text_strict_lowercases(self):
        result = clean_text_strict("HELLO World", self.stop_words)
        assert result == "hello world"

    def test_clean_text_strict_removes_stopwords(self):
        result = clean_text_strict("the quick brown fox", self.stop_words)
        assert "the" not in result

    def test_clean_text_light_preserves_spam_symbols(self):
        result = clean_text_light("Win $1000! Call now?")
        assert "$" in result
        assert "!" in result

    def test_clean_text_light_handles_empty_string(self):
        result = clean_text_light("")
        assert result == ""

    def test_clean_text_strict_handles_non_string(self):
        result = clean_text_strict(123, self.stop_words)
        assert result == ""

    def test_clean_text_strict_removes_short_words(self):
        result = clean_text_strict("a an are is", self.stop_words)
        assert "a" not in result
        assert "an" not in result

    def test_clean_text_light_removes_extra_spaces(self):
        result = clean_text_light("hello    world")
        assert "  " not in result


class TestCleanPipeline:

    @patch("data.clean.os.path.exists")
    def test_load_raw_data_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent.txt")

    @patch("data.clean.pd.read_csv")
    @patch("data.clean.os.path.exists")
    def test_load_raw_data_success(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["ham", "spam"],
            "text": ["hello", "win money"]
        })
        
        result = load_raw_data("test.txt")
        
        assert len(result) == 2
        assert "label" in result.columns
        assert "text" in result.columns

    def test_process_dataframe(self):
        df = pd.DataFrame({
            "text": ["Hello World!", "WIN $1000 NOW", "How are you?"]
        })
        
        result = process_dataframe(df)
        
        assert "clean_strict" in result.columns
        assert "clean_light" in result.columns
        assert len(result) == 3

    @patch("data.clean.process_dataframe")
    @patch("data.clean.load_raw_data")
    @patch("data.clean.pd.DataFrame.to_csv")
    @patch("data.clean.os.makedirs")
    def test_run_preprocessing_success(self, mock_makedirs, mock_to_csv, mock_load, mock_process):
        mock_load.return_value = pd.DataFrame({"label": ["ham"], "text": ["hello"]})
        mock_process.return_value = pd.DataFrame({"label": ["ham"], "text": ["hello"], "clean_strict": ["hello"], "clean_light": ["hello"]})
        
        run_preprocessing()
        
        mock_load.assert_called_once()
        mock_process.assert_called_once()
        mock_to_csv.assert_called_once()
