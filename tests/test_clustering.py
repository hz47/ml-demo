import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.clustering import SMSSpamDetector


class TestSMSSpamDetector:

    @pytest.fixture
    def mock_qdrant_client(self):
        with patch("llm.clustering.QdrantClient") as mock:
            yield mock.return_value

    def test_initialization(self):
        with patch("llm.clustering.QdrantClient"):
            detector = SMSSpamDetector()
            assert detector.collection_name == "sms_collection"
            assert detector.vector_name == "sms_embedding"
            assert detector.payload_key == "clean_light"

    def test_fetch_data_empty(self):
        with patch("llm.clustering.QdrantClient") as mock_client:
            mock_client.return_value.scroll.return_value = ([], None)
            
            detector = SMSSpamDetector()
            result = detector.fetch_data(max_points=10)
            
            assert result is False

    def test_fetch_data_success(self):
        with patch("llm.clustering.QdrantClient") as mock_client:
            mock_point = MagicMock()
            mock_point.payload = {"clean_light": "test message"}
            mock_point.vector = {"sms_embedding": [0.1] * 128}
            
            mock_client.return_value.scroll.return_value = ([mock_point], None)
            
            detector = SMSSpamDetector()
            result = detector.fetch_data(max_points=10)
            
            assert result is True
            assert detector.data is not None

    def test_run_spam_clustering(self):
        with patch("llm.clustering.QdrantClient"):
            detector = SMSSpamDetector()
            detector.data = pd.DataFrame({
                "SMS": ["hello world"] * 50,
                "clean_light": ["hello world"] * 50
            })
            detector.embeddings = np.random.randn(50, 10)
            
            detector.run_spam_clustering()
            
            assert "Cluster" in detector.data.columns
            assert "Label" in detector.data.columns
            assert "Confidence" in detector.data.columns

    @patch("llm.clustering.plt")
    def test_visualize(self, mock_plt):
        with patch("llm.clustering.QdrantClient"):
            with patch("llm.clustering.TSNE") as mock_tsne:
                mock_tsne_instance = MagicMock()
                mock_tsne.return_value = mock_tsne_instance
                mock_tsne_instance.fit_transform.return_value = np.random.randn(20, 2)
                
                detector = SMSSpamDetector()
                detector.data = pd.DataFrame({
                    "SMS": ["test"] * 20,
                    "Label": ["spam"] * 10 + ["ham"] * 10,
                    "Cluster": [0] * 10 + [1] * 10
                })
                detector.embeddings = np.random.randn(20, 10)
                
                detector.visualize()
                
                mock_plt.figure.assert_called()
                mock_plt.savefig.assert_called()
                mock_plt.close.assert_called()
