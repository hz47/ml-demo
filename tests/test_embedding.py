import pytest
import os
from unittest.mock import patch, MagicMock

from llm.embedding import split_into_batches, ensure_collection_exists, get_openai_client, get_qdrant_client


class TestEmbedding:

    def test_split_into_batches_basic(self):
        result = list(split_into_batches([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_split_into_batches_empty_list(self):
        result = list(split_into_batches([], 2))
        assert result == []

    def test_split_into_batches_exact_size(self):
        result = list(split_into_batches([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_split_into_batches_larger_batch(self):
        result = list(split_into_batches([1, 2, 3, 4, 5], 10))
        assert result == [[1, 2, 3, 4, 5]]

    def test_ensure_collection_creates_when_not_exists(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        
        ensure_collection_exists(mock_qdrant, "test_collection")
        
        mock_qdrant.collection_exists.assert_called_once_with("test_collection")
        mock_qdrant.create_collection.assert_called_once()

    def test_ensure_collection_skips_when_exists(self):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        
        ensure_collection_exists(mock_qdrant, "test_collection")
        
        mock_qdrant.create_collection.assert_not_called()

    @patch("llm.embedding.OpenAI")
    @patch("llm.embedding.QdrantClient")
    def test_get_clients_returns_both(self, mock_qdrant, mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_API_KEY": "test-qdrant"}):
            client = get_openai_client()
            qdrant = get_qdrant_client()
            
        assert client is not None
        assert qdrant is not None
