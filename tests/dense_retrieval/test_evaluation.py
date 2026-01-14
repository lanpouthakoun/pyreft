"""Unit tests for evaluation metrics module."""

import pytest
from dense_retrieval_pipeline.evaluation.metrics import RetrievalMetrics, RetrievalEvaluator


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics class."""

    def test_precision_at_k_all_relevant(self):
        """Test precision when all retrieved docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3", "doc4", "doc5"}
        
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 5) == 1.0
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 3) == 1.0

    def test_precision_at_k_none_relevant(self):
        """Test precision when no retrieved docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 3) == 0.0

    def test_precision_at_k_partial(self):
        """Test precision with partial relevance."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3"}
        
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 4) == 0.5
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 2) == 0.5

    def test_precision_at_k_zero_k(self):
        """Test precision with k=0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}
        
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 0) == 0.0

    def test_recall_at_k_all_found(self):
        """Test recall when all relevant docs are found."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2"}
        
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 5) == 1.0

    def test_recall_at_k_none_found(self):
        """Test recall when no relevant docs are found."""
        retrieved = ["doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2"}
        
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 3) == 0.0

    def test_recall_at_k_partial(self):
        """Test recall with partial retrieval."""
        retrieved = ["doc1", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 3) == 0.5

    def test_recall_at_k_empty_relevant(self):
        """Test recall with empty relevant set."""
        retrieved = ["doc1", "doc2"]
        relevant = set()
        
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 2) == 0.0

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        
        assert RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        retrieved = ["doc2", "doc1", "doc3"]
        relevant = {"doc1"}
        
        assert RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_mrr_third_position(self):
        """Test MRR when relevant doc is third."""
        retrieved = ["doc2", "doc3", "doc1"]
        relevant = {"doc1"}
        
        assert RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1/3)

    def test_mrr_not_found(self):
        """Test MRR when no relevant doc is found."""
        retrieved = ["doc2", "doc3", "doc4"]
        relevant = {"doc1"}
        
        assert RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_dcg_at_k_basic(self):
        """Test DCG calculation."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance_scores = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, 3)
        assert dcg > 0

    def test_dcg_at_k_zero_relevance(self):
        """Test DCG with zero relevance."""
        retrieved = ["doc1", "doc2"]
        relevance_scores = {"doc3": 1.0}
        
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, 2)
        assert dcg == 0.0

    def test_ndcg_at_k_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance_scores = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        
        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance_scores, 3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_at_k_worst_ranking(self):
        """Test NDCG with reversed ranking."""
        retrieved = ["doc3", "doc2", "doc1"]
        relevance_scores = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        
        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance_scores, 3)
        assert 0 < ndcg < 1.0

    def test_f1_at_k_perfect(self):
        """Test F1 with perfect precision and recall."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        
        f1 = RetrievalMetrics.f1_at_k(retrieved, relevant, 2)
        assert f1 == 1.0

    def test_f1_at_k_zero(self):
        """Test F1 with zero precision and recall."""
        retrieved = ["doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        
        f1 = RetrievalMetrics.f1_at_k(retrieved, relevant, 2)
        assert f1 == 0.0


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator class."""

    def test_evaluate_query(self):
        """Test single query evaluation."""
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
        
        result = evaluator.evaluate_query(
            query_id="q1",
            retrieved=["doc1", "doc2", "doc3", "doc4", "doc5"],
            relevant={"doc1", "doc3"}
        )
        
        assert result["query_id"] == "q1"
        assert "precision@1" in result
        assert "recall@3" in result
        assert "mrr" in result

    def test_evaluate_batch(self):
        """Test batch query evaluation."""
        evaluator = RetrievalEvaluator(k_values=[1, 5])
        
        queries = [
            {
                "query_id": "q1",
                "retrieved": ["doc1", "doc2"],
                "relevant": {"doc1"}
            },
            {
                "query_id": "q2",
                "retrieved": ["doc3", "doc4"],
                "relevant": {"doc3", "doc4"}
            }
        ]
        
        results = evaluator.evaluate_batch(queries)
        
        assert len(results) == 2
        assert results[0]["query_id"] == "q1"
        assert results[1]["query_id"] == "q2"

    def test_get_aggregate_metrics(self):
        """Test aggregate metrics calculation."""
        evaluator = RetrievalEvaluator(k_values=[1, 5])
        
        evaluator.evaluate_query("q1", ["doc1"], {"doc1"})
        evaluator.evaluate_query("q2", ["doc2"], {"doc2"})
        
        aggregate = evaluator.get_aggregate_metrics()
        
        assert "mean_precision@1" in aggregate
        assert "mean_mrr" in aggregate
        assert aggregate["num_queries"] == 2

    def test_reset(self):
        """Test evaluator reset."""
        evaluator = RetrievalEvaluator()
        
        evaluator.evaluate_query("q1", ["doc1"], {"doc1"})
        assert len(evaluator.results) == 1
        
        evaluator.reset()
        assert len(evaluator.results) == 0
