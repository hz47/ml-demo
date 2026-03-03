import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.utils import (
    SMSMetadataExtractor,
    SMSMetadataExtractorNB,
    load_data,
    find_high_precision_threshold,
)


class TestSMSMetadataExtractor:

    def test_fit_returns_self(self):
        extractor = SMSMetadataExtractor()
        result = extractor.fit(["text1", "text2"])
        assert result is extractor

    def test_transform_extracts_features(self):
        extractor = SMSMetadataExtractor()
        texts = ["Hello world", "WIN FREE MONEY!!!", "Hello"]
        result = extractor.transform(texts)
        
        assert result.shape[0] == 3
        assert result.shape[1] == 4

    def test_transform_counts_triggers(self):
        extractor = SMSMetadataExtractor()
        texts = ["win prize", "hello world"]
        result = extractor.transform(texts)
        
        assert result[0, 2] > 0
        assert result[1, 2] == 0

    def test_transform_counts_exclamations(self):
        extractor = SMSMetadataExtractor()
        texts = ["Hello!!!", "Hello"]
        result = extractor.transform(texts)
        
        assert result[0, 3] == 3
        assert result[1, 3] == 0


class TestSMSMetadataExtractorNB:

    def test_fit_returns_self(self):
        extractor = SMSMetadataExtractorNB()
        result = extractor.fit(["text1", "text2"])
        assert result is extractor

    def test_transform_extracts_features(self):
        extractor = SMSMetadataExtractorNB()
        texts = ["Hello world", "WIN FREE MONEY", "Hello"]
        result = extractor.transform(texts)
        
        assert result.shape[0] == 3
        assert result.shape[1] == 4

    def test_transform_handles_empty_string(self):
        extractor = SMSMetadataExtractorNB()
        texts = [""]
        result = extractor.transform(texts)
        
        assert result.shape[0] == 1
        assert result[0, 0] == 0

    def test_transform_calculates_sentence_count(self):
        extractor = SMSMetadataExtractorNB()
        texts = ["Hello. How are you? I'm fine!"]
        result = extractor.transform(texts)
        
        assert result[0, 2] > 0


class TestLoadData:

    @patch("ml.utils.pd.read_csv")
    def test_load_data_success(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "label": ["spam", "ham"],
            "text": ["test1", "test2"]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_data()
        
        assert len(result) == 2

    @patch("ml.utils.pd.read_csv")
    def test_load_data_drops_na(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "label": ["spam", "ham", None],
            "text": ["test1", "test2", "test3"]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_data()
        
        assert len(result) == 2


class TestFindHighPrecisionThreshold:

    def test_returns_threshold_when_precision_met(self):
        y_true = np.array(["spam", "spam", "ham", "ham"])
        probs = np.array([0.9, 0.8, 0.3, 0.2])
        
        result = find_high_precision_threshold(y_true, probs, target_precision=0.5)
        
        assert 0.0 <= result <= 1.0

    def test_returns_highest_valid_threshold(self):
        y_true = np.array(["spam", "spam", "ham", "ham", "spam"])
        probs = np.array([0.95, 0.85, 0.4, 0.3, 0.9])
        
        result = find_high_precision_threshold(y_true, probs, target_precision=0.8)
        
        assert result >= 0.85
