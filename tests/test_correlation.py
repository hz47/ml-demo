import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.correlation import compute_correlation_matrix, compute_correlation_by_label, run_correlation


class TestCorrelation:

    @patch("analysis.correlation.pd.read_csv")
    def test_compute_correlation_matrix(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham", "spam", "ham"],
            "text": ["short text", "longer text here", "another longer text", "tiny"]
        })
        
        result = compute_correlation_matrix()
        
        assert "text_length" in result.columns
        assert "word_count" in result.columns
        assert "sentence_count" in result.columns

    @patch("analysis.correlation.pd.read_csv")
    def test_compute_correlation_by_label(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham", "spam", "ham"],
            "text": ["short text", "longer text here", "another longer text", "tiny"]
        })
        
        result = compute_correlation_by_label()
        
        assert "spam" in result
        assert "ham" in result

    @patch("analysis.correlation.pd.read_csv")
    def test_run_correlation(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham"],
            "text": ["text1", "text2"]
        })
        
        result = run_correlation()
        
        assert result is not None
