import pytest
import os
from unittest.mock import patch, MagicMock
<<<<<<< HEAD

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
=======
from pathlib import Path

from llm.embedding import (
    ensure_collection_exists,
    generate_deterministic_id,
    load_sms_data,
    sync_sms_to_vector_store,
    search_similar_sms,
)


class TestEnsureCollection:
    @patch("llm.embedding.QdrantClient")
    def test_creates_when_not_exists(self, mock_qdrant_cls):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        mock_qdrant_cls.return_value = mock_qdrant

        ensure_collection_exists(mock_qdrant, "sms_collection")

        mock_qdrant.collection_exists.assert_called_once_with("sms_collection")
        mock_qdrant.create_collection.assert_called_once()

    @patch("llm.embedding.QdrantClient")
    def test_skips_when_exists(self, mock_qdrant_cls):
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant_cls.return_value = mock_qdrant

        ensure_collection_exists(mock_qdrant, "sms_collection")

        mock_qdrant.create_collection.assert_not_called()


class TestGenerateDeterministicId:
    def test_same_text_same_id(self):
        id1 = generate_deterministic_id("hello world")
        id2 = generate_deterministic_id("hello world")
        assert id1 == id2

    def test_different_text_different_id(self):
        id1 = generate_deterministic_id("hello world")
        id2 = generate_deterministic_id("different text")
        assert id1 != id2

    def test_id_length(self):
        id1 = generate_deterministic_id("hello world")
        assert len(id1) == 32


class TestLoadSmsData:
    @patch("llm.embedding.os.path.exists")
    @patch("llm.embedding.pd.read_csv")
    def test_loads_csv(self, mock_read_csv, mock_exists):
        mock_exists.return_value = True
        mock_df = MagicMock()
        mock_df.dropna.return_value = mock_df
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(to_dict=MagicMock(return_value=[{"a": 1}])))
        mock_read_csv.return_value = mock_df

        result = load_sms_data("dummy_path.csv")

        mock_exists.assert_called_once_with("dummy_path.csv")
        mock_read_csv.assert_called_once_with("dummy_path.csv")
        assert isinstance(result, list)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_sms_data("/nonexistent/path.csv")


class TestSyncSmsToVectorStore:
    @patch("llm.embedding.QdrantClient")
    @patch("llm.embedding.OpenAI")
    @patch("llm.embedding.models")
    def test_sync_records(self, mock_models, mock_openai_cls, mock_qdrant_cls):
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.embeddings.create.return_value = mock_response

        records = [
            {"text": "test1", "clean_light": "clean1", "label": "ham"},
            {"text": "test2", "clean_light": "clean2", "label": "spam"},
        ]
        sync_sms_to_vector_store(mock_openai, mock_qdrant, records)

        mock_openai.embeddings.create.assert_called_once()
        mock_qdrant.upsert.assert_called_once()


class TestSearchSimilarSms:
    @patch("llm.embedding.QdrantClient")
    @patch("llm.embedding.OpenAI")
    def test_search(self, mock_openai_cls, mock_qdrant_cls):
        mock_openai = MagicMock()
        mock_openai_cls.return_value = mock_openai

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.embeddings.create.return_value = mock_response

        mock_result = MagicMock()
        mock_result.payload = {"label": "ham", "text": "hello"}
        mock_result.score = 0.95
        mock_qdrant.query_points.return_value.points = [mock_result]

        results = search_similar_sms(mock_openai, mock_qdrant, "hello world")

        mock_openai.embeddings.create.assert_called_once()
        mock_qdrant.query_points.assert_called_once()
        assert len(results) == 1
>>>>>>> 49c38dd062e56690fa2dc74ece3adcf68b13335b
