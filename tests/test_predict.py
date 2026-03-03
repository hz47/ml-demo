import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPredictSMS:

    @patch("ml.predict.joblib.load")
    @patch("ml.predict.clean_text_v3")
    def test_predict_sms_spam(self, mock_clean, mock_load):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = [[0.2, 0.8]]
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        
        mock_load.return_value = {
            "model": mock_pipeline,
            "threshold": 0.5
        }
        mock_clean.return_value = "win free money"
        
        import importlib
        import ml.predict
        importlib.reload(ml.predict)
        
        result = ml.predict.predict_sms("Win $1000 now!")
        
        assert result == "spam"

    @patch("ml.predict.joblib.load")
    @patch("ml.predict.clean_text_v3")
    def test_predict_sms_ham(self, mock_clean, mock_load):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = [[0.9, 0.1]]
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        
        mock_load.return_value = {
            "model": mock_pipeline,
            "threshold": 0.5
        }
        mock_clean.return_value = "hello how are you"
        
        import importlib
        import ml.predict
        importlib.reload(ml.predict)
        
        result = ml.predict.predict_sms("Hello, how are you?")
        
        assert result == "ham"

    @patch("ml.predict.joblib.load")
    @patch("ml.predict.clean_text_v3")
    def test_predict_sms_at_threshold(self, mock_clean, mock_load):
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = [[0.5, 0.5]]
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        
        mock_load.return_value = {
            "model": mock_pipeline,
            "threshold": 0.5
        }
        mock_clean.return_value = "test message"
        
        import importlib
        import ml.predict
        importlib.reload(ml.predict)
        
        result = ml.predict.predict_sms("test")
        
        assert result == "spam"
