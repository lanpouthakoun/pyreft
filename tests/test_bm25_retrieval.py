"""Unit tests for BM25 retrieval module."""

import pytest
from src.bm25_retrieval.retriever import BM25Retriever


class TestBM25Retriever:
    """Tests for BM25Retriever class."""

    def test_initialization(self):
        """Test BM25Retriever initialization."""
        retriever = BM25Retriever(k1=1.5, b=0.75)
        
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.bm25 is None
        assert retriever.documents == []

    def test_preprocess_basic(self):
        """Test basic text preprocessing."""
        retriever = BM25Retriever()
        
        tokens = retriever._preprocess("Hello World!")
        assert tokens == ["hello", "world"]

    def test_preprocess_punctuation(self):
        """Test preprocessing removes punctuation."""
        retriever = BM25Retriever()
        
        tokens = retriever._preprocess("Hello, World! How are you?")
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_preprocess_multiple_spaces(self):
        """Test preprocessing handles multiple spaces."""
        retriever = BM25Retriever()
        
        tokens = retriever._preprocess("Hello    World")
        assert tokens == ["hello", "world"]

    def test_preprocess_case_insensitive(self):
        """Test preprocessing converts to lowercase."""
        retriever = BM25Retriever()
        
        tokens = retriever._preprocess("HELLO World HeLLo")
        assert tokens == ["hello", "world", "hello"]

    def test_index_documents(self):
        """Test document indexing."""
        retriever = BM25Retriever()
        documents = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "A quick brown dog"
        ]
        
        retriever.index_documents(documents)
        
        assert len(retriever.documents) == 3
        assert len(retriever.document_ids) == 3
        assert retriever.bm25 is not None

    def test_index_documents_with_ids(self):
        """Test document indexing with custom IDs."""
        retriever = BM25Retriever()
        documents = ["doc one", "doc two"]
        doc_ids = ["id_1", "id_2"]
        
        retriever.index_documents(documents, doc_ids)
        
        assert retriever.document_ids == ["id_1", "id_2"]

    def test_search_basic(self):
        """Test basic search functionality."""
        retriever = BM25Retriever()
        documents = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "A quick brown dog"
        ]
        
        retriever.index_documents(documents)
        results = retriever.search("quick brown", top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_search_returns_scores(self):
        """Test that search returns valid scores."""
        retriever = BM25Retriever()
        documents = [
            "artificial intelligence machine learning",
            "natural language processing",
            "computer vision deep learning"
        ]
        
        retriever.index_documents(documents)
        results = retriever.search("machine learning", top_k=3)
        
        assert results[0][2] > 0

    def test_search_ranking(self):
        """Test that search results are ranked by relevance."""
        retriever = BM25Retriever()
        documents = [
            "artificial intelligence",
            "machine learning artificial intelligence",
            "natural language processing"
        ]
        doc_ids = ["doc_0", "doc_1", "doc_2"]
        
        retriever.index_documents(documents, doc_ids)
        results = retriever.search("artificial intelligence", top_k=3)
        
        scores = [r[2] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_without_index_raises_error(self):
        """Test that search without indexing raises error."""
        retriever = BM25Retriever()
        
        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test query")

    def test_batch_search(self):
        """Test batch search functionality."""
        retriever = BM25Retriever()
        documents = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "A quick brown dog"
        ]
        
        retriever.index_documents(documents)
        queries = ["quick brown", "lazy dog"]
        results = retriever.batch_search(queries, top_k=2)
        
        assert len(results) == 2
        assert all(len(r) == 2 for r in results)

    def test_get_model_info(self):
        """Test model info retrieval."""
        retriever = BM25Retriever(k1=1.2, b=0.8)
        documents = ["doc one", "doc two", "doc three"]
        
        retriever.index_documents(documents)
        info = retriever.get_model_info()
        
        assert info["model_name"] == "BM25Okapi"
        assert info["model_type"] == "sparse"
        assert info["k1"] == 1.2
        assert info["b"] == 0.8
        assert info["num_documents"] == 3
        assert info["vocabulary_size"] > 0
        assert info["avg_doc_length"] > 0

    def test_exact_match_ranking(self):
        """Test that documents with query terms rank higher than those without."""
        retriever = BM25Retriever()
        documents = [
            "cloud computing services",
            "unrelated topic about cooking recipes",
            "cloud computing infrastructure"
        ]
        doc_ids = ["doc_0", "doc_1", "doc_2"]
        
        retriever.index_documents(documents, doc_ids)
        results = retriever.search("cloud computing", top_k=3)
        
        top_doc_id = results[0][0]
        assert top_doc_id in ["doc_0", "doc_2"]
        assert results[2][0] == "doc_1"

    def test_empty_query(self):
        """Test search with empty query."""
        retriever = BM25Retriever()
        documents = ["doc one", "doc two"]
        
        retriever.index_documents(documents)
        results = retriever.search("", top_k=2)
        
        assert len(results) == 2
        assert all(r[2] == 0.0 for r in results)

    def test_query_not_in_corpus(self):
        """Test search with query terms not in corpus."""
        retriever = BM25Retriever()
        documents = ["apple banana cherry", "dog cat mouse"]
        
        retriever.index_documents(documents)
        results = retriever.search("xyz123", top_k=2)
        
        assert len(results) == 2
        assert all(r[2] == 0.0 for r in results)
