import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train_nb import SMSMetadataExtractor, run_training


class TestTrainNB:

    def test_sms_metadata_extractor_fit(self):
        extractor = SMSMetadataExtractor()
        result = extractor.fit(["text1", "text2"])
        assert result is extractor

    def test_sms_metadata_extractor_transform(self):
        extractor = SMSMetadataExtractor()
        texts = ["Hello world", "WIN PRIZE NOW!!!", "test"]
        result = extractor.transform(texts)
        
        assert result.shape[0] == 3
        assert result.shape[1] == 4

    def test_sms_metadata_extractor_counts_triggers(self):
        extractor = SMSMetadataExtractor()
        texts = ["win free prize", "hello world"]
        result = extractor.transform(texts)
        
        assert result[0, 3] > 0
        assert result[1, 3] == 0

    def test_sms_metadata_extractor_sentence_count(self):
        extractor = SMSMetadataExtractor()
        texts = ["Hello! How are you? I'm fine!"]
        result = extractor.transform(texts)
        
        assert result[0, 2] >= 3


class TestRunTraining:

    @patch("ml.train_nb.load_data")
    @patch("ml.train_nb.train_test_split")
    @patch("ml.train_nb.joblib.dump")
    @patch("ml.train_nb.os.makedirs")
    def test_run_training_returns_dict(self, mock_makedirs, mock_dump, mock_split, mock_load):
        mock_df = pd.DataFrame({
            "text": ["spam message"] * 10 + ["ham message"] * 10,
            "clean_light": ["spam"] * 10 + ["ham"] * 10,
            "label": ["spam"] * 10 + ["ham"] * 10
        })
        mock_load.return_value = mock_df
        
        X_train = pd.DataFrame({"text": ["test"] * 14, "clean_light": ["test"] * 14})
        X_val = pd.DataFrame({"text": ["test"] * 3, "clean_light": ["test"] * 3})
        X_test = pd.DataFrame({"text": ["test"] * 3, "clean_light": ["test"] * 3})
        y_train = pd.Series(["spam"] * 7 + ["ham"] * 7)
        y_val = pd.Series(["spam"] * 2 + ["ham"] * 1)
        y_test = pd.Series(["spam"] * 1 + ["ham"] * 2)
        
        mock_split.side_effect = [
            (X_train, X_test, y_train, y_test),
            (X_train, X_val, y_train, y_val)
        ]
        
        mock_pipeline = MagicMock()
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        
        with patch("ml.train_nb.Pipeline") as mock_pipe_cls:
            mock_pipe_cls.return_value = mock_pipeline
            result = run_training()
        
        assert "model" in result
        assert "threshold" in result
        assert result["model"] == "Naive Bayes"
