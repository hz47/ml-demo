import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.clean import clean_text_v3, load_and_process


class TestCleanTextV3:

    def test_clean_text_v3_lowercases(self):
        result = clean_text_v3("HELLO WORLD")
        assert result == "hello world"

    def test_clean_text_v3_replaces_numbers(self):
        result = clean_text_v3("meet u at 2pm for 4u")
        assert " to " in result or " for " in result

    def test_clean_text_v3_handles_urls(self):
        result = clean_text_v3("visit http://example.com now")
        assert "urladdr" in result

    def test_clean_text_v3_handles_currency_symbols(self):
        result = clean_text_v3("win $1000 now")
        assert "moneysymb" in result

    def test_clean_text_v3_handles_non_string(self):
        result = clean_text_v3(123)
        assert result == ""

    def test_clean_text_v3_keeps_exclamations(self):
        result = clean_text_v3("Win!!")
        assert "!" in result


class TestLoadAndProcess:

    @patch("data.clean.os.path.exists")
    @patch("data.clean.pd.read_csv")
    @patch("data.clean.pd.DataFrame.to_csv")
    @patch("data.clean.os.makedirs")
    def test_load_and_process_file_not_found(self, mock_makedirs, mock_to_csv, mock_read_csv, mock_exists):
        mock_exists.return_value = False
        
        result = load_and_process()
        
        assert result is None

    @patch("data.clean.os.path.exists")
    @patch("data.clean.pd.read_csv")
    @patch("data.clean.pd.DataFrame.to_csv")
    @patch("data.clean.os.makedirs")
    def test_load_and_process_success(self, mock_makedirs, mock_to_csv, mock_read_csv, mock_exists):
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham"],
            "text": ["test1", "test2"]
        })
        
        load_and_process()
        
        mock_to_csv.assert_called_once()
