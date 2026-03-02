import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train import calculate_best_threshold, prepare_data, create_pipeline, evaluate_model, run_model_training


class TestTrain:

    @pytest.fixture
    def dummy_dataframe(self):
        data = {
            "clean_light": [
                "win free money now", "hello how are you", "click here for prize",
                "meeting at 3pm", "congratulations you won", "what is the weather",
                "urgent call this number", "dinner tonight", "claim your reward now",
                "see you tomorrow", "special offer inside", "lunch at noon",
                "free trial today", "movie tonight", "limited time deal",
                "happy birthday", "urgent business matter", "winner notification",
                "call me later", "weekend plans"
            ],
            "label": [
                "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
                "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "ham", "ham"
            ]
        }
        return pd.DataFrame(data)

    def test_calculate_best_threshold(self):
        y_true = np.array(["spam", "spam", "ham", "ham"])
        probs = np.array([0.9, 0.8, 0.3, 0.2])
        
        threshold = calculate_best_threshold(y_true, probs)
        
        assert isinstance(threshold, (float, np.floating))
        assert 0.0 <= threshold <= 1.0

    def test_calculate_best_threshold_all_same_prob(self):
        y_true = np.array(["spam", "spam", "ham", "ham"])
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        
        threshold = calculate_best_threshold(y_true, probs)
        
        assert isinstance(threshold, (float, np.floating))

    @patch("ml.train.pd.read_csv")
    def test_prepare_data(self, mock_read_csv, dummy_dataframe):
        mock_read_csv.return_value = dummy_dataframe
        
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data("dummy_path.csv")
        
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_val) > 0
        assert len(y_test) > 0

    def test_create_pipeline(self):
        pipeline = create_pipeline()
        
        assert pipeline is not None
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")

    @patch("ml.train.pd.read_csv")
    def test_prepare_data_returns_correct_splits(self, mock_read_csv):
        data = {
            "clean_light": ["text"] * 20,
            "label": ["spam"] * 10 + ["ham"] * 10
        }
        mock_read_csv.return_value = pd.DataFrame(data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data("dummy_path.csv")
        
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == 20
        assert len(X_train) == 14
        assert len(X_val) == 3
        assert len(X_test) == 3

    def test_evaluate_model(self):
        mock_pipeline = MagicMock()
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        X_test = pd.Series(["hello world", "win money"])
        y_test = pd.Series(["ham", "spam"])
        threshold = 0.5
        
        with patch("builtins.print"):
            evaluate_model(mock_pipeline, X_test, y_test, threshold)
        
        mock_pipeline.predict_proba.assert_called()

    @patch("ml.train.prepare_data")
    @patch("ml.train.create_pipeline")
    @patch("ml.train.evaluate_model")
    @patch("ml.train.joblib.dump")
    @patch("ml.train.os.makedirs")
    def test_run_model_training(self, mock_makedirs, mock_dump, mock_eval, mock_create, mock_prepare):
        X_train = pd.Series(["text1", "text2", "text3", "text4"])
        X_val = pd.Series(["text5"])
        X_test = pd.Series(["text6"])
        y_train = pd.Series(["spam", "ham", "spam", "ham"])
        y_val = pd.Series(["spam"])
        y_test = pd.Series(["ham"])
        
        mock_prepare.return_value = (X_train, X_val, X_test, y_train, y_val, y_test)
        
        mock_pipeline = MagicMock()
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        
        def mock_predict_proba(X):
            n = len(X) if hasattr(X, '__len__') else 1
            probs = np.random.rand(n, 2)
            probs[:, 1] = probs[:, 1] * 0.3
            return probs
        
        mock_pipeline.predict_proba.side_effect = mock_predict_proba
        mock_create.return_value = mock_pipeline
        
        run_model_training()
        
        mock_prepare.assert_called_once()
        mock_create.assert_called_once()
        mock_pipeline.fit.assert_called_once()
        mock_dump.assert_called_once()
        mock_makedirs.assert_called_once()
