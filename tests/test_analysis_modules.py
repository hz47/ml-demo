import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.sms_length import get_length_stats_by_label, run_sms_length
from analysis.sms_word_count import get_avg_word_count_by_label, run_sms_word_count
from analysis.sms_sentence_count import get_avg_sentence_count_by_label, run_sms_sentence_count


class TestSmsLength:

    @patch("analysis.sms_length.pd.read_csv")
    def test_get_length_stats_by_label(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham", "spam", "ham"],
            "text": ["short", "a bit longer text here", "medium length", "tiny"]
        })
        
        result = get_length_stats_by_label()
        
        assert "spam" in result.index
        assert "ham" in result.index
        assert "mean" in result.columns
        assert "median" in result.columns

    @patch("analysis.sms_length.pd.read_csv")
    def test_run_sms_length(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham"],
            "text": ["text1", "text2"]
        })
        
        result = run_sms_length()
        
        assert result is not None


class TestSmsWordCount:

    @patch("analysis.sms_word_count.pd.read_csv")
    def test_get_avg_word_count_by_label(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham", "spam", "ham"],
            "text": ["one two three", "four five", "six seven eight nine", "ten"]
        })
        
        result = get_avg_word_count_by_label()
        
        assert "spam" in result.index
        assert "ham" in result.index

    @patch("analysis.sms_word_count.pd.read_csv")
    def test_run_sms_word_count(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham"],
            "text": ["text1", "text2"]
        })
        
        result = run_sms_word_count()
        
        assert result is not None


class TestSmsSentenceCount:

    @patch("analysis.sms_sentence_count.pd.read_csv")
    def test_get_avg_sentence_count_by_label(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham", "spam", "ham"],
            "text": ["Hello. How are you?", "Hi!", "One. Two. Three.", "Ok."]
        })
        
        result = get_avg_sentence_count_by_label()
        
        assert "spam" in result.index
        assert "ham" in result.index

    @patch("analysis.sms_sentence_count.pd.read_csv")
    def test_run_sms_sentence_count(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham"],
            "text": ["text1", "text2"]
        })
        
        result = run_sms_sentence_count()
        
        assert result is not None
