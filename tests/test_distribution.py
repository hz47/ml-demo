import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.distribution import get_label_distribution, run_distribution


class TestDistribution:

    @patch("analysis.distribution.pd.read_csv")
    def test_get_label_distribution(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham", "spam", "ham", "spam"]
        })
        
        result = get_label_distribution()
        
        assert result["spam"] == 3
        assert result["ham"] == 2

    @patch("analysis.distribution.pd.read_csv")
    def test_run_distribution(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            "label": ["spam", "ham"]
        })
        
        result = run_distribution()
        
        assert "spam" in result.index
        assert "ham" in result.index
