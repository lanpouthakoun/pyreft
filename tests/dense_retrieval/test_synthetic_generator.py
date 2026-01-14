"""Unit tests for synthetic document generator module."""

import pytest
from dense_retrieval_pipeline.synthetic_generator.generator import (
    SyntheticDocumentGenerator,
    SyntheticDataset,
    DocumentQueryPair
)


class TestDocumentQueryPair:
    """Tests for DocumentQueryPair dataclass."""

    def test_create_pair(self):
        """Test creating a document-query pair."""
        pair = DocumentQueryPair(
            doc_id="doc_1",
            document="Test document content",
            query="Test query",
            relevance=1.0,
            category="technology",
            semantic_type="exact_match"
        )
        
        assert pair.doc_id == "doc_1"
        assert pair.document == "Test document content"
        assert pair.query == "Test query"
        assert pair.relevance == 1.0
        assert pair.category == "technology"
        assert pair.semantic_type == "exact_match"


class TestSyntheticDataset:
    """Tests for SyntheticDataset dataclass."""

    def test_create_dataset(self):
        """Test creating a synthetic dataset."""
        dataset = SyntheticDataset(
            documents=["doc1", "doc2"],
            document_ids=["id1", "id2"],
            queries=["query1"],
            relevance_map={"q_0": {"id1": 1.0, "id2": 0.0}}
        )
        
        assert len(dataset.documents) == 2
        assert len(dataset.document_ids) == 2
        assert len(dataset.queries) == 1

    def test_get_relevant_docs(self):
        """Test getting relevant documents for a query."""
        dataset = SyntheticDataset(
            documents=["doc1", "doc2", "doc3"],
            document_ids=["id1", "id2", "id3"],
            queries=["query1"],
            relevance_map={"q_0": {"id1": 1.0, "id2": 0.5, "id3": 0.0}}
        )
        
        relevant = dataset.get_relevant_docs(0)
        assert "id1" in relevant
        assert "id2" in relevant
        assert "id3" not in relevant

    def test_get_relevant_docs_missing_query(self):
        """Test getting relevant docs for non-existent query."""
        dataset = SyntheticDataset(
            documents=["doc1"],
            document_ids=["id1"],
            queries=["query1"],
            relevance_map={}
        )
        
        relevant = dataset.get_relevant_docs(99)
        assert relevant == []


class TestSyntheticDocumentGenerator:
    """Tests for SyntheticDocumentGenerator class."""

    def test_init_with_seed(self):
        """Test generator initialization with seed."""
        generator = SyntheticDocumentGenerator(seed=42)
        assert generator.generated_pairs == []

    def test_generate_dataset_default(self):
        """Test generating dataset with default parameters."""
        generator = SyntheticDocumentGenerator(seed=42)
        dataset = generator.generate_dataset(num_pairs=10)
        
        assert len(dataset.documents) > 0
        assert len(dataset.queries) > 0
        assert len(dataset.document_ids) == len(dataset.documents)

    def test_generate_dataset_100_pairs(self):
        """Test generating 100+ document-query pairs."""
        generator = SyntheticDocumentGenerator(seed=42)
        dataset = generator.generate_dataset(num_pairs=100)
        
        assert len(dataset.queries) >= 100

    def test_generate_dataset_with_hard_negatives(self):
        """Test generating dataset with hard negatives."""
        generator = SyntheticDocumentGenerator(seed=42)
        dataset = generator.generate_dataset(
            num_pairs=20,
            include_hard_negatives=True
        )
        
        assert len(dataset.documents) > 20

    def test_generate_dataset_without_hard_negatives(self):
        """Test generating dataset without hard negatives."""
        generator = SyntheticDocumentGenerator(seed=42)
        dataset = generator.generate_dataset(
            num_pairs=20,
            include_hard_negatives=False
        )
        
        assert len(dataset.queries) >= 20

    def test_relevance_map_structure(self):
        """Test that relevance map has correct structure."""
        generator = SyntheticDocumentGenerator(seed=42)
        dataset = generator.generate_dataset(num_pairs=10)
        
        for query_key, doc_relevances in dataset.relevance_map.items():
            assert query_key.startswith("q_")
            for doc_id, relevance in doc_relevances.items():
                assert doc_id.startswith("doc_")
                assert 0.0 <= relevance <= 1.0

    def test_get_statistics(self):
        """Test getting generation statistics."""
        generator = SyntheticDocumentGenerator(seed=42)
        generator.generate_dataset(num_pairs=50)
        
        stats = generator.get_statistics()
        
        assert "total_pairs" in stats
        assert stats["total_pairs"] > 0
        assert "categories" in stats
        assert "semantic_types" in stats

    def test_get_statistics_empty(self):
        """Test getting statistics before generation."""
        generator = SyntheticDocumentGenerator()
        stats = generator.get_statistics()
        
        assert stats["total_pairs"] == 0

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        generator1 = SyntheticDocumentGenerator(seed=123)
        dataset1 = generator1.generate_dataset(num_pairs=10)
        
        generator2 = SyntheticDocumentGenerator(seed=123)
        dataset2 = generator2.generate_dataset(num_pairs=10)
        
        assert dataset1.documents == dataset2.documents
        assert dataset1.queries == dataset2.queries

    def test_topics_coverage(self):
        """Test that all topics are covered in generation."""
        generator = SyntheticDocumentGenerator(seed=42)
        generator.generate_dataset(num_pairs=100)
        
        stats = generator.get_statistics()
        categories = stats.get("categories", {})
        
        assert len(categories) >= 3

    def test_semantic_types_coverage(self):
        """Test that multiple semantic types are generated."""
        generator = SyntheticDocumentGenerator(seed=42)
        generator.generate_dataset(num_pairs=100, include_hard_negatives=True)
        
        stats = generator.get_statistics()
        semantic_types = stats.get("semantic_types", {})
        
        assert len(semantic_types) >= 2
