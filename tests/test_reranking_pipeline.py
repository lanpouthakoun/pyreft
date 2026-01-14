"""
Tests for the two-stage retrieval pipeline with cross-encoder reranking.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reranking_pipeline import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    EvaluationMetrics,
    QueryResult,
    RetrievalEvaluator,
    RetrievalResult,
    TwoStageRetrievalPipeline,
    analyze_latency_tradeoffs,
    create_synthetic_dataset,
    generate_recommendations,
)


class TestRetrievalResult:
    def test_retrieval_result_creation(self):
        result = RetrievalResult(
            doc_id="doc_1",
            text="Test document",
            score=0.95,
            rank=1
        )
        assert result.doc_id == "doc_1"
        assert result.text == "Test document"
        assert result.score == 0.95
        assert result.rank == 1


class TestQueryResult:
    def test_query_result_creation(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="Doc 1", score=0.9, rank=1),
            RetrievalResult(doc_id="doc_2", text="Doc 2", score=0.8, rank=2),
        ]
        query_result = QueryResult(
            query_id="q1",
            query_text="test query",
            results=results,
            retrieval_time_ms=10.5,
            reranking_time_ms=5.2
        )
        assert query_result.query_id == "q1"
        assert len(query_result.results) == 2
        assert query_result.retrieval_time_ms == 10.5


class TestEvaluationMetrics:
    def test_metrics_to_dict(self):
        metrics = EvaluationMetrics(
            mrr=0.85,
            ndcg_at_10=0.75,
            recall_at_10=0.60,
            avg_retrieval_time_ms=5.0,
            avg_reranking_time_ms=10.0,
            total_queries=100
        )
        result = metrics.to_dict()
        assert result["mrr"] == 0.85
        assert result["ndcg@10"] == 0.75
        assert result["recall@10"] == 0.60
        assert result["total_queries"] == 100


class TestBM25Retriever:
    @pytest.fixture
    def sample_documents(self):
        return [
            {"id": "doc_0", "text": "machine learning is a subset of artificial intelligence"},
            {"id": "doc_1", "text": "deep learning uses neural networks"},
            {"id": "doc_2", "text": "natural language processing handles text data"},
            {"id": "doc_3", "text": "computer vision processes images and videos"},
            {"id": "doc_4", "text": "reinforcement learning learns from rewards"},
        ]

    def test_bm25_name(self):
        retriever = BM25Retriever()
        assert retriever.name == "BM25"

    def test_bm25_index_and_retrieve(self, sample_documents):
        retriever = BM25Retriever()
        retriever.index(sample_documents)
        
        results, time_ms = retriever.retrieve("machine learning artificial intelligence", top_k=3)
        
        assert len(results) == 3
        assert time_ms > 0
        assert results[0].doc_id == "doc_0"
        assert results[0].rank == 1

    def test_bm25_retrieve_without_index_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="Index not built"):
            retriever.retrieve("test query")

    def test_bm25_tokenize(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]


class TestDenseRetriever:
    @pytest.fixture
    def sample_documents(self):
        return [
            {"id": "doc_0", "text": "machine learning is a subset of artificial intelligence"},
            {"id": "doc_1", "text": "deep learning uses neural networks"},
            {"id": "doc_2", "text": "natural language processing handles text data"},
        ]

    def test_dense_name(self):
        retriever = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert "MiniLM" in retriever.name

    def test_dense_retrieve_without_index_raises(self):
        retriever = DenseRetriever()
        with pytest.raises(ValueError, match="Index not built"):
            retriever.retrieve("test query")

    @patch("sentence_transformers.SentenceTransformer")
    def test_dense_index_and_retrieve_mocked(self, mock_st_class, sample_documents):
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            np.array([[0.1, 0.2, 0.3]]),
        ]
        
        retriever = DenseRetriever()
        retriever.index(sample_documents)
        
        results, time_ms = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
        assert time_ms > 0


class TestCrossEncoderReranker:
    def test_reranker_name(self):
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert "MiniLM" in reranker.name

    def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        results, time_ms = reranker.rerank("test query", [], top_k=10)
        assert results == []
        assert time_ms == 0.0

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_mocked(self, mock_ce_class):
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])
        
        candidates = [
            RetrievalResult(doc_id="doc_0", text="Doc 0", score=0.5, rank=1),
            RetrievalResult(doc_id="doc_1", text="Doc 1", score=0.4, rank=2),
            RetrievalResult(doc_id="doc_2", text="Doc 2", score=0.3, rank=3),
        ]
        
        reranker = CrossEncoderReranker()
        results, time_ms = reranker.rerank("test query", candidates, top_k=2)
        
        assert len(results) == 2
        assert results[0].doc_id == "doc_1"
        assert results[0].rank == 1
        assert results[1].doc_id == "doc_2"


class TestTwoStageRetrievalPipeline:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.name = "MockRetriever"
        retriever.retrieve.return_value = (
            [
                RetrievalResult(doc_id="doc_0", text="Doc 0", score=0.9, rank=1),
                RetrievalResult(doc_id="doc_1", text="Doc 1", score=0.8, rank=2),
            ],
            5.0
        )
        return retriever

    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()
        reranker.name = "MockReranker"
        reranker.rerank.return_value = (
            [
                RetrievalResult(doc_id="doc_1", text="Doc 1", score=0.95, rank=1),
                RetrievalResult(doc_id="doc_0", text="Doc 0", score=0.85, rank=2),
            ],
            10.0
        )
        return reranker

    def test_pipeline_name_without_reranker(self, mock_retriever):
        pipeline = TwoStageRetrievalPipeline(mock_retriever, reranker=None)
        assert pipeline.name == "MockRetriever"

    def test_pipeline_name_with_reranker(self, mock_retriever, mock_reranker):
        pipeline = TwoStageRetrievalPipeline(mock_retriever, reranker=mock_reranker)
        assert "MockRetriever" in pipeline.name
        assert "MockReranker" in pipeline.name

    def test_pipeline_retrieve_without_reranker(self, mock_retriever):
        pipeline = TwoStageRetrievalPipeline(
            mock_retriever, reranker=None, first_stage_k=100, final_k=10
        )
        result = pipeline.retrieve("test query", "q1")
        
        assert result.query_id == "q1"
        assert result.retrieval_time_ms == 5.0
        assert result.reranking_time_ms == 0.0

    def test_pipeline_retrieve_with_reranker(self, mock_retriever, mock_reranker):
        pipeline = TwoStageRetrievalPipeline(
            mock_retriever, reranker=mock_reranker, first_stage_k=100, final_k=10
        )
        result = pipeline.retrieve("test query", "q1")
        
        assert result.query_id == "q1"
        assert result.retrieval_time_ms == 5.0
        assert result.reranking_time_ms == 10.0
        assert result.results[0].doc_id == "doc_1"


class TestRetrievalEvaluator:
    def test_compute_mrr_hit_at_1(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
            RetrievalResult(doc_id="doc_2", text="", score=0.8, rank=2),
        ]
        mrr = RetrievalEvaluator.compute_mrr(results, {"doc_1"})
        assert mrr == 1.0

    def test_compute_mrr_hit_at_2(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
            RetrievalResult(doc_id="doc_2", text="", score=0.8, rank=2),
        ]
        mrr = RetrievalEvaluator.compute_mrr(results, {"doc_2"})
        assert mrr == 0.5

    def test_compute_mrr_no_hit(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
            RetrievalResult(doc_id="doc_2", text="", score=0.8, rank=2),
        ]
        mrr = RetrievalEvaluator.compute_mrr(results, {"doc_3"})
        assert mrr == 0.0

    def test_compute_dcg(self):
        relevances = [1.0, 1.0, 0.0, 1.0]
        dcg = RetrievalEvaluator.compute_dcg(relevances, k=4)
        expected = 1.0 + 1.0/np.log2(3) + 0.0 + 1.0/np.log2(5)
        assert abs(dcg - expected) < 1e-6

    def test_compute_dcg_empty(self):
        dcg = RetrievalEvaluator.compute_dcg([], k=10)
        assert dcg == 0.0

    def test_compute_ndcg_perfect(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
            RetrievalResult(doc_id="doc_2", text="", score=0.8, rank=2),
        ]
        ndcg = RetrievalEvaluator.compute_ndcg(results, {"doc_1", "doc_2"}, k=10)
        assert ndcg == 1.0

    def test_compute_ndcg_no_relevant(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
        ]
        ndcg = RetrievalEvaluator.compute_ndcg(results, set(), k=10)
        assert ndcg == 0.0

    def test_compute_recall(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
            RetrievalResult(doc_id="doc_2", text="", score=0.8, rank=2),
        ]
        recall = RetrievalEvaluator.compute_recall(results, {"doc_1", "doc_3"}, k=10)
        assert recall == 0.5

    def test_compute_recall_empty_relevant(self):
        results = [
            RetrievalResult(doc_id="doc_1", text="", score=0.9, rank=1),
        ]
        recall = RetrievalEvaluator.compute_recall(results, set(), k=10)
        assert recall == 0.0


class TestSyntheticDataset:
    def test_create_synthetic_dataset(self):
        documents, queries, qrels = create_synthetic_dataset(num_queries=10, num_docs=100)
        
        assert len(documents) == 100
        assert len(queries) == 10
        assert len(qrels) == 10
        
        for doc in documents:
            assert "id" in doc
            assert "text" in doc
        
        for query in queries:
            assert "id" in query
            assert "text" in query
        
        for qid, relevant_docs in qrels.items():
            assert isinstance(relevant_docs, set)
            assert len(relevant_docs) > 0


class TestLatencyAnalysis:
    def test_analyze_latency_tradeoffs(self):
        results = {
            "bm25_only": {
                "mrr": 0.8,
                "ndcg@10": 0.75,
                "avg_retrieval_time_ms": 5.0
            },
            "dense_only": {
                "mrr": 0.85,
                "ndcg@10": 0.80,
                "avg_retrieval_time_ms": 50.0
            },
            "bm25_with_reranking": {
                "k=20": {
                    "mrr": 0.9,
                    "ndcg@10": 0.85,
                    "avg_retrieval_time_ms": 5.0,
                    "avg_reranking_time_ms": 30.0
                }
            },
            "dense_with_reranking": {}
        }
        
        analysis = analyze_latency_tradeoffs(results)
        
        assert "bm25_baseline" in analysis
        assert "dense_baseline" in analysis
        assert "reranking_overhead" in analysis
        assert "accuracy_improvements" in analysis
        
        assert analysis["bm25_baseline"]["latency_ms"] == 5.0
        assert "bm25_k=20" in analysis["reranking_overhead"]


class TestRecommendations:
    def test_generate_recommendations(self):
        results = {
            "bm25_only": {"mrr": 0.8},
            "dense_only": {"mrr": 0.85},
            "bm25_with_reranking": {
                "k=20": {"mrr": 0.9},
                "k=50": {"mrr": 0.92},
                "k=100": {"mrr": 0.93}
            }
        }
        
        latency_analysis = {
            "reranking_overhead": {
                "bm25_k=20": {"reranking_overhead_ms": 30.0},
                "bm25_k=50": {"reranking_overhead_ms": 60.0},
                "bm25_k=100": {"reranking_overhead_ms": 120.0}
            },
            "accuracy_improvements": {
                "bm25_k=20": {"mrr_improvement": 0.1},
                "bm25_k=50": {"mrr_improvement": 0.12},
                "bm25_k=100": {"mrr_improvement": 0.13}
            }
        }
        
        recommendations = generate_recommendations(results, latency_analysis)
        
        assert "best_accuracy_config" in recommendations
        assert "best_mrr" in recommendations
        assert "optimal_candidates_to_rerank" in recommendations


class TestIntegration:
    def test_full_pipeline_with_bm25(self):
        documents = [
            {"id": f"doc_{i}", "text": f"Document about topic {i % 5} with content {i}"}
            for i in range(50)
        ]
        
        retriever = BM25Retriever()
        retriever.index(documents)
        
        pipeline = TwoStageRetrievalPipeline(
            retriever, reranker=None, first_stage_k=10, final_k=5
        )
        
        result = pipeline.retrieve("topic 0 content", "q1")
        
        assert len(result.results) == 5
        assert result.retrieval_time_ms > 0
        assert result.reranking_time_ms == 0.0

    def test_evaluator_with_pipeline(self):
        documents = [
            {"id": f"doc_{i}", "text": f"Document about topic {i % 3}"}
            for i in range(30)
        ]
        queries = [
            {"id": "q0", "text": "topic 0"},
            {"id": "q1", "text": "topic 1"},
        ]
        qrels = {
            "q0": {f"doc_{i}" for i in range(30) if i % 3 == 0},
            "q1": {f"doc_{i}" for i in range(30) if i % 3 == 1},
        }
        
        retriever = BM25Retriever()
        retriever.index(documents)
        
        pipeline = TwoStageRetrievalPipeline(
            retriever, reranker=None, first_stage_k=20, final_k=10
        )
        
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate(pipeline, queries, qrels)
        
        assert metrics.total_queries == 2
        assert 0 <= metrics.mrr <= 1
        assert 0 <= metrics.ndcg_at_10 <= 1
        assert 0 <= metrics.recall_at_10 <= 1
