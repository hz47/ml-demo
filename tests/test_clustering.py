import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.clustering import LargeSMSClusterer, ClusterLabel


class TestClustering:

    @pytest.fixture
    def mock_qdrant_client(self):
        with patch("llm.clustering.QdrantClient") as mock:
            yield mock.return_value

    @pytest.fixture
    def clusterer_with_mock(self, mock_qdrant_client):
        with patch("llm.clustering.QdrantClient"):
            return LargeSMSClusterer()

    def test_cluster_label_pydantic_model(self):
        label = ClusterLabel(
            category_name="Finance",
            primary_tone="Urgent",
            justification="Contains money-related keywords"
        )
        assert label.category_name == "Finance"
        assert label.primary_tone == "Urgent"

    def test_clusterer_initialization(self, mock_qdrant_client):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer(
                collection_name="test_collection",
                random_state=123
            )
            assert clusterer.collection_name == "test_collection"
            assert clusterer.random_state == 123

    def test_maybe_pca_with_sufficient_samples(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.embeddings_norm = np.random.randn(100, 200)
            
            clusterer.maybe_pca(n_components=50)
            
            assert clusterer.pca_model is not None
            assert clusterer.embeddings_norm.shape[1] <= 50

    def test_maybe_pca_with_insufficient_samples(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.embeddings_norm = np.random.randn(1, 5)
            
            clusterer.maybe_pca(n_components=50)
            
            assert clusterer.pca_model is None

    def test_run_clustering_returns_score(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer(random_state=42)
            clusterer.data = pd.DataFrame({"SMS": ["hello world"] * 50})
            clusterer.embeddings_norm = np.random.randn(50, 10)
            
            score = clusterer.run_clustering(k_min=3, k_max=5)
            
            assert isinstance(score, (float, np.floating))
            assert "Cluster" in clusterer.data.columns

    def test_generate_cluster_labels_parses_response(self):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_parsed = ClusterLabel(
            category_name="Spam",
            primary_tone="Urgent",
            justification="Contains spam keywords"
        )
        mock_completion.choices[0].message.parsed = mock_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.data = pd.DataFrame({
                "SMS": ["win money now"] * 20,
                "Cluster": [0] * 20
            })
            clusterer.embeddings_norm = np.random.randn(20, 10)

            result = clusterer.generate_cluster_labels(mock_client)

            assert 0 in result
            assert result[0] == "Spam"

    def test_fetch_data_success(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            
            mock_point = MagicMock()
            mock_point.payload = {"clean": "test message"}
            mock_point.vector = {"sms_embedding": [0.1] * 128}
            
            clusterer.client.scroll.return_value = ([mock_point], None)
            
            result = clusterer.fetch_data(max_points=100)
            
            assert result is True
            assert clusterer.data is not None
            assert len(clusterer.data) == 1

    def test_fetch_data_no_points(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.client.scroll.return_value = ([], None)
            
            result = clusterer.fetch_data()
            
            assert result is False

    def test_fetch_data_no_valid_pairs(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            
            mock_point = MagicMock()
            mock_point.payload = {}
            mock_point.vector = {}
            
            clusterer.client.scroll.return_value = ([mock_point], None)
            
            result = clusterer.fetch_data()
            
            assert result is False

    def test_get_cluster_insights(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.data = pd.DataFrame({
                "SMS": ["hello world", "test message"],
                "Cluster": [0, 1]
            })
            
            with patch("builtins.print") as mock_print:
                clusterer.get_cluster_insights(top_n=2)
                
                assert mock_print.called

    def test_export_results(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.data = pd.DataFrame({
                "SMS": ["hello", "world"],
                "Cluster": [0, 1]
            })
            
            with patch("pandas.DataFrame.to_csv") as mock_csv:
                clusterer.export_results("test_output.csv")
                
                mock_csv.assert_called_once_with("test_output.csv", index=False, encoding="utf-8-sig")

    def test_visualize_tsne(self):
        with patch("llm.clustering.QdrantClient"):
            clusterer = LargeSMSClusterer()
            clusterer.data = pd.DataFrame({
                "SMS": ["test"] * 100,
                "Cluster": [0] * 100
            })
            clusterer.embeddings_norm = np.random.randn(100, 50)
            
            with patch("llm.clustering.TSNE") as mock_tsne:
                with patch("llm.clustering.plt") as mock_plt:
                    mock_tsne_instance = MagicMock()
                    mock_tsne.return_value = mock_tsne_instance
                    mock_tsne_instance.fit_transform.return_value = np.random.randn(100, 2)
                    
                    clusterer.visualize_tsne(max_points=50)
                    
                    mock_tsne.assert_called_once()
