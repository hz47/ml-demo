import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.benchmark import (
    load_test_data,
    load_model,
    predict_with_threshold,
    calculate_metrics,
    measure_inference_time,
    determine_winner,
)


class TestBenchmark:

    @patch("steps.benchmark.pd.read_csv")
    def test_load_test_data(self, mock_read_csv):
        data = {
            "clean_light": ["text1", "text2", "text3", "text4"],
            "label": ["spam", "ham", "spam", "ham"]
        }
        mock_read_csv.return_value = pd.DataFrame(data)

        X_test, y_test = load_test_data()

        assert len(X_test) > 0
        assert len(y_test) > 0

    @patch("steps.benchmark.joblib.load")
    def test_load_model(self, mock_joblib):
        mock_artifacts = {
            "pipeline": MagicMock(),
            "threshold": 0.5,
            "classes": np.array(["ham", "spam"])
        }
        mock_joblib.return_value = mock_artifacts

        pipeline, threshold, classes = load_model("dummy_path.pkl")

        assert pipeline is not None
        assert threshold == 0.5
        assert list(classes) == ["ham", "spam"]

    def test_predict_with_threshold(self):
        mock_pipeline = MagicMock()
        mock_pipeline.classes_ = np.array(["ham", "spam"])
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

        X = pd.Series(["text1", "text2"])
        predictions = predict_with_threshold(mock_pipeline, X, 0.5)

        assert len(predictions) == 2

    def test_calculate_metrics(self):
        y_true = np.array(["spam", "spam", "ham", "ham"])
        y_pred = np.array(["spam", "spam", "ham", "ham"])

        metrics = calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision_spam" in metrics
        assert "recall_spam" in metrics
        assert "f1_spam" in metrics
        assert metrics["accuracy"] == 1.0

    def test_calculate_metrics_with_errors(self):
        y_true = np.array(["spam", "spam", "ham", "ham"])
        y_pred = np.array(["spam", "ham", "spam", "ham"])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.5
        assert "f1_macro" in metrics

    def test_measure_inference_time(self):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array(["ham", "spam"])

        X = pd.Series(["text1", "text2"])
        mean_time, std_time = measure_inference_time(mock_pipeline, X, n_runs=3)

        assert mean_time >= 0
        assert std_time >= 0
        assert mock_pipeline.predict.call_count == 3

    def test_determine_winner_higher_is_better(self):
        assert determine_winner("accuracy", 0.9, 0.8) == "LR"
        assert determine_winner("accuracy", 0.8, 0.9) == "NB"
        assert determine_winner("accuracy", 0.5, 0.5) == "TIE"

    def test_determine_winner_lower_is_better(self):
        assert determine_winner("inference_time", 0.001, 0.002, lower_is_better=True) == "LR"
        assert determine_winner("inference_time", 0.002, 0.001, lower_is_better=True) == "NB"
        assert determine_winner("inference_time", 0.5, 0.5, lower_is_better=True) == "TIE"
