"""
Tests for the hybrid retrieval pipeline.

This module contains unit tests for BM25 retrieval, dense retrieval,
score fusion methods, and evaluation metrics.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from pyreft.hybrid_pipeline import (
    Document,
    RetrievalResult,
    BM25Retriever,
    DenseRetriever,
    ScoreFusion,
    EvaluationMetrics,
    HybridRetriever,
    create_sample_dataset,
    run_evaluation,
)


class TestDocument:
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        doc = Document(doc_id="test_1", text="This is a test document")
        assert doc.doc_id == "test_1"
        assert doc.text == "This is a test document"
        assert doc.metadata == {}
    
    def test_document_with_metadata(self):
        doc = Document(
            doc_id="test_2",
            text="Another document",
            metadata={"source": "test", "score": 0.5}
        )
        assert doc.metadata["source"] == "test"
        assert doc.metadata["score"] == 0.5


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        result = RetrievalResult(doc_id="doc_1", score=0.95, rank=1)
        assert result.doc_id == "doc_1"
        assert result.score == 0.95
        assert result.rank == 1
        assert result.text is None
    
    def test_retrieval_result_with_text(self):
        result = RetrievalResult(
            doc_id="doc_2",
            score=0.8,
            rank=2,
            text="Document text"
        )
        assert result.text == "Document text"


class TestBM25Retriever:
    """Tests for BM25 retrieval."""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("doc_0", "machine learning algorithms"),
            Document("doc_1", "deep learning neural networks"),
            Document("doc_2", "natural language processing"),
            Document("doc_3", "machine learning models"),
            Document("doc_4", "computer vision systems"),
        ]
    
    @pytest.fixture
    def bm25_retriever(self, sample_documents):
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(sample_documents)
        return retriever
    
    def test_bm25_initialization(self):
        retriever = BM25Retriever(k1=1.2, b=0.8, epsilon=0.3)
        assert retriever.k1 == 1.2
        assert retriever.b == 0.8
        assert retriever.epsilon == 0.3
    
    def test_bm25_index(self, sample_documents):
        retriever = BM25Retriever()
        retriever.index(sample_documents)
        
        assert retriever._indexed is True
        assert len(retriever.corpus) == 5
        assert len(retriever.doc_lengths) == 5
        assert retriever.avgdl > 0
    
    def test_bm25_retrieve_without_index(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve("test query")
    
    def test_bm25_retrieve(self, bm25_retriever):
        results = bm25_retriever.retrieve("machine learning", top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        assert results[0].score >= results[1].score >= results[2].score
    
    def test_bm25_retrieve_relevant_docs_ranked_higher(self, bm25_retriever):
        results = bm25_retriever.retrieve("machine learning", top_k=5)
        
        top_doc_ids = [r.doc_id for r in results[:2]]
        assert "doc_0" in top_doc_ids or "doc_3" in top_doc_ids
    
    def test_bm25_tokenization(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Hello World TEST")
        assert tokens == ["hello", "world", "test"]
    
    def test_bm25_idf_calculation(self, bm25_retriever):
        assert "machine" in bm25_retriever.idf
        assert "learning" in bm25_retriever.idf
        assert all(v > 0 for v in bm25_retriever.idf.values())


class TestDenseRetriever:
    """Tests for dense retrieval."""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("doc_0", "machine learning algorithms"),
            Document("doc_1", "deep learning neural networks"),
            Document("doc_2", "natural language processing"),
        ]
    
    def test_dense_initialization(self):
        retriever = DenseRetriever(
            model_name="all-MiniLM-L6-v2",
            similarity_fn="cosine"
        )
        assert retriever.model_name == "all-MiniLM-L6-v2"
        assert retriever.similarity_fn == "cosine"
    
    def test_dense_retrieve_without_index(self):
        retriever = DenseRetriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve("test query")
    
    @patch.object(DenseRetriever, '_get_model')
    def test_dense_index(self, mock_get_model, sample_documents):
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_get_model.return_value = mock_model
        
        retriever = DenseRetriever()
        retriever.index(sample_documents)
        
        assert retriever._indexed is True
        assert len(retriever.corpus) == 3
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape[0] == 3
    
    @patch.object(DenseRetriever, '_get_model')
    def test_dense_retrieve(self, mock_get_model, sample_documents):
        import numpy as np
        
        mock_model = MagicMock()
        doc_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        query_embedding = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        
        mock_model.encode.side_effect = [doc_embeddings, query_embedding]
        mock_get_model.return_value = mock_model
        
        retriever = DenseRetriever()
        retriever.index(sample_documents)
        results = retriever.retrieve("machine learning", top_k=3)
        
        assert len(results) == 3
        assert results[0].rank == 1
        assert results[0].doc_id == "doc_0"


class TestScoreFusion:
    """Tests for score fusion methods."""
    
    @pytest.fixture
    def bm25_results(self):
        return [
            RetrievalResult("doc_0", 10.0, 1, "text 0"),
            RetrievalResult("doc_1", 8.0, 2, "text 1"),
            RetrievalResult("doc_2", 5.0, 3, "text 2"),
            RetrievalResult("doc_3", 3.0, 4, "text 3"),
        ]
    
    @pytest.fixture
    def dense_results(self):
        return [
            RetrievalResult("doc_2", 0.95, 1, "text 2"),
            RetrievalResult("doc_0", 0.85, 2, "text 0"),
            RetrievalResult("doc_4", 0.75, 3, "text 4"),
            RetrievalResult("doc_1", 0.65, 4, "text 1"),
        ]
    
    def test_normalize_scores(self, bm25_results):
        normalized = ScoreFusion.normalize_scores(bm25_results)
        
        assert len(normalized) == len(bm25_results)
        assert normalized[0].score == 1.0
        assert normalized[-1].score == 0.0
        assert all(0 <= r.score <= 1 for r in normalized)
    
    def test_normalize_scores_empty(self):
        normalized = ScoreFusion.normalize_scores([])
        assert normalized == []
    
    def test_normalize_scores_same_values(self):
        results = [
            RetrievalResult("doc_0", 5.0, 1),
            RetrievalResult("doc_1", 5.0, 2),
        ]
        normalized = ScoreFusion.normalize_scores(results)
        assert all(r.score == 1.0 for r in normalized)
    
    def test_reciprocal_rank_fusion(self, bm25_results, dense_results):
        combined = ScoreFusion.reciprocal_rank_fusion(
            [bm25_results, dense_results],
            k=60,
            top_k=5
        )
        
        assert len(combined) == 5
        assert combined[0].rank == 1
        assert all(isinstance(r, RetrievalResult) for r in combined)
        
        doc_ids = {r.doc_id for r in combined}
        assert "doc_0" in doc_ids
        assert "doc_2" in doc_ids
    
    def test_rrf_score_calculation(self):
        results1 = [RetrievalResult("doc_0", 1.0, 1)]
        results2 = [RetrievalResult("doc_0", 1.0, 1)]
        
        combined = ScoreFusion.reciprocal_rank_fusion(
            [results1, results2],
            k=60,
            top_k=1
        )
        
        expected_score = 2 * (1.0 / (60 + 1))
        assert abs(combined[0].score - expected_score) < 1e-6
    
    def test_weighted_combination(self, bm25_results, dense_results):
        combined = ScoreFusion.weighted_combination(
            bm25_results,
            dense_results,
            alpha=0.5,
            top_k=5
        )
        
        assert len(combined) == 5
        assert combined[0].rank == 1
        assert all(isinstance(r, RetrievalResult) for r in combined)
    
    def test_weighted_combination_alpha_1(self, bm25_results, dense_results):
        combined = ScoreFusion.weighted_combination(
            bm25_results,
            dense_results,
            alpha=1.0,
            top_k=4
        )
        
        assert combined[0].doc_id == bm25_results[0].doc_id
    
    def test_weighted_combination_alpha_0(self, bm25_results, dense_results):
        combined = ScoreFusion.weighted_combination(
            bm25_results,
            dense_results,
            alpha=0.0,
            top_k=4
        )
        
        assert combined[0].doc_id == dense_results[0].doc_id


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    @pytest.fixture
    def results_list(self):
        return [
            [
                RetrievalResult("doc_0", 0.9, 1),
                RetrievalResult("doc_1", 0.8, 2),
                RetrievalResult("doc_2", 0.7, 3),
            ],
            [
                RetrievalResult("doc_3", 0.9, 1),
                RetrievalResult("doc_4", 0.8, 2),
                RetrievalResult("doc_5", 0.7, 3),
            ],
        ]
    
    @pytest.fixture
    def binary_relevance(self):
        return {
            "0": ["doc_0", "doc_2"],
            "1": ["doc_5"],
        }
    
    @pytest.fixture
    def graded_relevance(self):
        return {
            "0": {"doc_0": 3, "doc_2": 2, "doc_1": 1},
            "1": {"doc_5": 3, "doc_4": 1},
        }
    
    def test_mrr_perfect(self):
        results = [
            [RetrievalResult("doc_0", 1.0, 1)],
            [RetrievalResult("doc_1", 1.0, 1)],
        ]
        relevance = {"0": ["doc_0"], "1": ["doc_1"]}
        
        mrr = EvaluationMetrics.mean_reciprocal_rank(results, relevance)
        assert mrr == 1.0
    
    def test_mrr_second_position(self):
        results = [
            [
                RetrievalResult("doc_x", 1.0, 1),
                RetrievalResult("doc_0", 0.9, 2),
            ],
        ]
        relevance = {"0": ["doc_0"]}
        
        mrr = EvaluationMetrics.mean_reciprocal_rank(results, relevance)
        assert mrr == 0.5
    
    def test_mrr_no_relevant(self):
        results = [
            [RetrievalResult("doc_x", 1.0, 1)],
        ]
        relevance = {"0": ["doc_0"]}
        
        mrr = EvaluationMetrics.mean_reciprocal_rank(results, relevance)
        assert mrr == 0.0
    
    def test_mrr_with_fixture(self, results_list, binary_relevance):
        mrr = EvaluationMetrics.mean_reciprocal_rank(results_list, binary_relevance)
        expected = (1.0 + 1/3) / 2
        assert abs(mrr - expected) < 1e-6
    
    def test_ndcg_perfect(self):
        results = [
            [
                RetrievalResult("doc_0", 1.0, 1),
                RetrievalResult("doc_1", 0.9, 2),
            ],
        ]
        relevance = {"0": {"doc_0": 3, "doc_1": 2}}
        
        ndcg = EvaluationMetrics.ndcg_at_k(results, relevance, k=2)
        assert ndcg == 1.0
    
    def test_ndcg_reversed(self):
        results = [
            [
                RetrievalResult("doc_1", 1.0, 1),
                RetrievalResult("doc_0", 0.9, 2),
            ],
        ]
        relevance = {"0": {"doc_0": 3, "doc_1": 2}}
        
        ndcg = EvaluationMetrics.ndcg_at_k(results, relevance, k=2)
        assert ndcg < 1.0
    
    def test_recall_at_k_perfect(self):
        results = [
            [
                RetrievalResult("doc_0", 1.0, 1),
                RetrievalResult("doc_1", 0.9, 2),
            ],
        ]
        relevance = {"0": ["doc_0", "doc_1"]}
        
        recall = EvaluationMetrics.recall_at_k(results, relevance, k=2)
        assert recall == 1.0
    
    def test_recall_at_k_partial(self):
        results = [
            [
                RetrievalResult("doc_0", 1.0, 1),
                RetrievalResult("doc_x", 0.9, 2),
            ],
        ]
        relevance = {"0": ["doc_0", "doc_1"]}
        
        recall = EvaluationMetrics.recall_at_k(results, relevance, k=2)
        assert recall == 0.5
    
    def test_recall_at_k_none(self):
        results = [
            [
                RetrievalResult("doc_x", 1.0, 1),
                RetrievalResult("doc_y", 0.9, 2),
            ],
        ]
        relevance = {"0": ["doc_0", "doc_1"]}
        
        recall = EvaluationMetrics.recall_at_k(results, relevance, k=2)
        assert recall == 0.0


class TestHybridRetriever:
    """Tests for hybrid retriever."""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("doc_0", "machine learning algorithms"),
            Document("doc_1", "deep learning neural networks"),
            Document("doc_2", "natural language processing"),
        ]
    
    def test_hybrid_initialization(self):
        retriever = HybridRetriever(
            bm25_params={"k1": 1.2, "b": 0.8},
            fusion_method="rrf",
            fusion_params={"k": 30}
        )
        
        assert retriever.bm25.k1 == 1.2
        assert retriever.bm25.b == 0.8
        assert retriever.fusion_method == "rrf"
        assert retriever.fusion_params["k"] == 30
    
    def test_hybrid_retrieve_without_index(self):
        retriever = HybridRetriever()
        with pytest.raises(RuntimeError, match="Index not built"):
            retriever.retrieve("test query")
    
    @patch.object(DenseRetriever, '_get_model')
    def test_hybrid_index(self, mock_get_model, sample_documents):
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_get_model.return_value = mock_model
        
        retriever = HybridRetriever()
        retriever.index(sample_documents)
        
        assert retriever._indexed is True
        assert retriever.bm25._indexed is True
        assert retriever.dense._indexed is True
    
    @patch.object(DenseRetriever, '_get_model')
    def test_hybrid_retrieve_rrf(self, mock_get_model, sample_documents):
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.random.rand(3, 384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        mock_get_model.return_value = mock_model
        
        retriever = HybridRetriever(fusion_method="rrf")
        retriever.index(sample_documents)
        results = retriever.retrieve("machine learning", top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    @patch.object(DenseRetriever, '_get_model')
    def test_hybrid_retrieve_weighted(self, mock_get_model, sample_documents):
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.random.rand(3, 384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        mock_get_model.return_value = mock_model
        
        retriever = HybridRetriever(
            fusion_method="weighted",
            fusion_params={"alpha": 0.7}
        )
        retriever.index(sample_documents)
        results = retriever.retrieve("machine learning", top_k=3)
        
        assert len(results) == 3
    
    @patch.object(DenseRetriever, '_get_model')
    def test_hybrid_retrieve_return_individual(self, mock_get_model, sample_documents):
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.random.rand(3, 384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        mock_get_model.return_value = mock_model
        
        retriever = HybridRetriever()
        retriever.index(sample_documents)
        hybrid, bm25, dense = retriever.retrieve(
            "machine learning",
            top_k=3,
            return_individual=True
        )
        
        assert len(hybrid) == 3
        assert len(bm25) == 3
        assert len(dense) == 3
    
    def test_hybrid_invalid_fusion_method(self, sample_documents):
        retriever = HybridRetriever(fusion_method="invalid")
        retriever.bm25.index(sample_documents)
        retriever.dense._indexed = True
        retriever.dense.corpus = sample_documents
        retriever.dense.embeddings = None
        retriever._indexed = True
        
        with pytest.raises(ValueError, match="Unknown fusion method"):
            retriever.bm25.retrieve = MagicMock(return_value=[])
            retriever.dense.retrieve = MagicMock(return_value=[])
            retriever.retrieve("test")


class TestCreateSampleDataset:
    """Tests for sample dataset creation."""
    
    def test_create_sample_dataset(self):
        docs, queries, binary_rel, graded_rel = create_sample_dataset()
        
        assert len(docs) == 20
        assert len(queries) == 5
        assert len(binary_rel) == 5
        assert len(graded_rel) == 5
        
        assert all(isinstance(d, Document) for d in docs)
        assert all(isinstance(q, str) for q in queries)
        
        for query_id in binary_rel:
            assert query_id in graded_rel
            for doc_id in binary_rel[query_id]:
                assert doc_id in graded_rel[query_id]


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @patch.object(DenseRetriever, '_get_model')
    def test_full_pipeline(self, mock_get_model):
        import numpy as np
        
        docs, queries, binary_rel, graded_rel = create_sample_dataset()
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda x, **kwargs: (
            np.random.rand(len(x) if isinstance(x, list) else 1, 384).astype(np.float32)
            if isinstance(x, list)
            else np.random.rand(384).astype(np.float32)
        )
        mock_get_model.return_value = mock_model
        
        retriever = HybridRetriever(fusion_method="rrf", fusion_params={"k": 60})
        retriever.index(docs)
        
        results = run_evaluation(retriever, queries, binary_rel, graded_rel)
        
        assert "bm25" in results
        assert "dense" in results
        assert "hybrid" in results
        
        for method in ["bm25", "dense", "hybrid"]:
            assert "mrr" in results[method]
            assert "ndcg@10" in results[method]
            assert "recall@10" in results[method]
            assert "recall@100" in results[method]
            
            assert 0 <= results[method]["mrr"] <= 1
            assert 0 <= results[method]["ndcg@10"] <= 1
            assert 0 <= results[method]["recall@10"] <= 1
            assert 0 <= results[method]["recall@100"] <= 1
    
    @patch.object(DenseRetriever, '_get_model')
    def test_tuning(self, mock_get_model):
        import numpy as np
        
        docs, queries, binary_rel, _ = create_sample_dataset()
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda x, **kwargs: (
            np.random.rand(len(x) if isinstance(x, list) else 1, 384).astype(np.float32)
            if isinstance(x, list)
            else np.random.rand(384).astype(np.float32)
        )
        mock_get_model.return_value = mock_model
        
        retriever = HybridRetriever()
        retriever.index(docs)
        
        tuning_results = retriever.tune_fusion_weights(
            queries[:2],
            binary_rel,
            metric="mrr",
            alpha_range=[0.0, 0.5, 1.0],
            rrf_k_range=[30, 60]
        )
        
        assert "weighted" in tuning_results
        assert "rrf" in tuning_results
        assert "best_weighted" in tuning_results
        assert "best_rrf" in tuning_results
        
        assert len(tuning_results["weighted"]) == 3
        assert len(tuning_results["rrf"]) == 2
        
        assert "alpha" in tuning_results["best_weighted"]
        assert "score" in tuning_results["best_weighted"]
        assert "k" in tuning_results["best_rrf"]
        assert "score" in tuning_results["best_rrf"]
