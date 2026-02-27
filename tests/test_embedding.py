import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.embedding import chunked, ensure_collection


class TestEmbedding:

    def test_chunked_basic(self):
        result = list(chunked([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunked_empty_list(self):
        result = list(chunked([], 2))
        assert result == []

    def test_chunked_exact_size(self):
        result = list(chunked([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_chunked_larger_batch(self):
        result = list(chunked([1, 2, 3, 4, 5], 10))
        assert result == [[1, 2, 3, 4, 5]]

    def test_ensure_collection_creates_when_not_exists(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        
        ensure_collection(mock_qdrant, "test_collection")
        
        mock_qdrant.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant.create_collection.assert_called_once()

    def test_ensure_collection_skips_when_exists(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        
        ensure_collection(mock_qdrant, "test_collection")
        
        mock_qdrant.create_collection.assert_not_called()

    @patch("llm.embedding.OpenAI")
    @patch("llm.embedding.QdrantClient")
    def test_get_clients_returns_both(self, mock_qdrant, mock_openai):
        from llm.embedding import get_clients
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_API_KEY": "test-qdrant"}):
            client, qdrant = get_clients()
            
        assert client is not None
        assert qdrant is not None
