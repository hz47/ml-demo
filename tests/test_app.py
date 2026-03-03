import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestAPI:

    @patch("app.os.path.exists")
    def test_health_endpoint(self, mock_exists):
        mock_exists.return_value = False
        from app import app
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["svm_loaded"] is False

    @patch("app.os.path.exists")
    @patch("app.joblib.load")
    def test_health_endpoint_with_model(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_pipeline = MagicMock()
        mock_pipeline.classes_ = ["ham", "spam"]
        mock_load.return_value = {
            "model": mock_pipeline,
            "threshold": 0.5
        }
        from app import app
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["svm_loaded"] is True

    @patch("app.os.path.exists")
    @patch("app.joblib.load")
    def test_predict_spam(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]
        mock_pipeline.classes_ = ["ham", "spam"]
        mock_load.return_value = {
            "model": mock_pipeline,
            "threshold": 0.5
        }
        from app import app
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "Win $1000 now!"})
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == "spam"

    @patch("app.os.path.exists")
    @patch("app.joblib.load")
    def test_predict_ham(self, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = [[0.9, 0.1]]
        mock_pipeline.classes_ = ["ham", "spam"]
        mock_load.return_value = {
            "model": mock_pipeline,
            "threshold": 0.5
        }
        from app import app
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "Hello, how are you?"})
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == "ham"

    def test_predict_invalid_input(self):
        from app import app
        with TestClient(app) as client:
            response = client.post("/predict", json={"wrong_field": "value"})
            assert response.status_code == 422
