import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestAPI:

    def test_health_endpoint(self):
        with patch("app.joblib.load") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]
            mock_load.return_value = {
                "pipeline": mock_pipeline,
                "threshold": 0.5,
                "classes": ["ham", "spam"]
            }
            from app import app
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}

    def test_home_endpoint(self):
        with patch("app.joblib.load") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]
            mock_load.return_value = {
                "pipeline": mock_pipeline,
                "threshold": 0.5,
                "classes": ["ham", "spam"]
            }
            from app import app
            with TestClient(app) as client:
                response = client.get("/")
                assert response.status_code == 200
                data = response.json()
                assert "message" in data
                assert "threshold" in data

    def test_predict_spam(self):
        with patch("app.joblib.load") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]
            mock_load.return_value = {
                "pipeline": mock_pipeline,
                "threshold": 0.5,
                "classes": ["ham", "spam"]
            }
            from app import app
            with TestClient(app) as client:
                response = client.post("/predict", json={"text": "Win $1000 now!"})
                assert response.status_code == 200
                data = response.json()
                assert data["prediction"] == "spam"

    def test_predict_ham(self):
        with patch("app.joblib.load") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = [[0.9, 0.1]]
            mock_load.return_value = {
                "pipeline": mock_pipeline,
                "threshold": 0.5,
                "classes": ["ham", "spam"]
            }
            from app import app
            with TestClient(app) as client:
                response = client.post("/predict", json={"text": "Hello, how are you?"})
                assert response.status_code == 200
                data = response.json()
                assert data["prediction"] == "ham"

    def test_predict_invalid_input(self):
        with patch("app.joblib.load") as mock_load:
            mock_pipeline = MagicMock()
            mock_pipeline.predict_proba.return_value = [[0.1, 0.9]]
            mock_load.return_value = {
                "pipeline": mock_pipeline,
                "threshold": 0.5,
                "classes": ["ham", "spam"]
            }
            from app import app
            with TestClient(app) as client:
                response = client.post("/predict", json={"wrong_field": "value"})
                assert response.status_code == 422
