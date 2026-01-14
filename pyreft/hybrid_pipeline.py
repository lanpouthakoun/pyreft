"""
Hybrid Retrieval Pipeline combining BM25 and Dense Retrieval.

This module implements a hybrid retrieval system that combines sparse (BM25) and
dense (embedding-based) retrieval methods with various score fusion strategies.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class Document:
    """Represents a document in the retrieval system."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    doc_id: str
    score: float
    rank: int
    text: Optional[str] = None


@dataclass
class QueryResult:
    """Represents results for a single query."""
    query_id: str
    query_text: str
    results: List[RetrievalResult]


class BM25Retriever:
    """BM25 sparse retrieval implementation."""
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
            epsilon: Floor for IDF values (default 0.25)
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus: List[Document] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self._indexed = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def index(self, documents: List[Document]) -> None:
        """
        Index a corpus of documents.
        
        Args:
            documents: List of Document objects to index
        """
        self.corpus = documents
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}
        
        for doc in documents:
            tokens = self._tokenize(doc.text)
            self.doc_lengths.append(len(tokens))
            
            term_freqs: Dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            self.doc_term_freqs.append(term_freqs)
            
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        n_docs = len(documents)
        self.idf = {}
        for term, df in self.doc_freqs.items():
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = max(idf, self.epsilon)
        
        self._indexed = True
    
    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a single document."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        for token in query_tokens:
            if token not in term_freqs:
                continue
            
            tf = term_freqs[token]
            idf = self.idf.get(token, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(
        self,
        query: str,
        top_k: int = 100
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by score descending
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            score = self._score_document(query_tokens, idx)
            scores.append((doc.doc_id, score, doc.text))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score, text) in enumerate(scores[:top_k], start=1):
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                text=text
            ))
        
        return results


class DenseRetriever:
    """Dense retrieval using embedding similarity."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_fn: str = "cosine"
    ):
        """
        Initialize dense retriever.
        
        Args:
            model_name: Name of the sentence-transformers model
            similarity_fn: Similarity function ('cosine' or 'dot')
        """
        self.model_name = model_name
        self.similarity_fn = similarity_fn
        self.corpus: List[Document] = []
        self.embeddings: Optional[NDArray[np.float32]] = None
        self._model = None
        self._indexed = False
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for dense retrieval. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def _compute_similarity(
        self,
        query_embedding: NDArray[np.float32],
        doc_embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute similarity between query and documents."""
        if self.similarity_fn == "cosine":
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            return np.dot(doc_norms, query_norm)
        elif self.similarity_fn == "dot":
            return np.dot(doc_embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_fn}")
    
    def index(self, documents: List[Document]) -> None:
        """
        Index a corpus of documents.
        
        Args:
            documents: List of Document objects to index
        """
        self.corpus = documents
        model = self._get_model()
        
        texts = [doc.text for doc in documents]
        self.embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self._indexed = True
    
    def retrieve(
        self,
        query: str,
        top_k: int = 100
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by score descending
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")
        
        model = self._get_model()
        query_embedding = model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        
        similarities = self._compute_similarity(query_embedding, self.embeddings)
        
        scored_docs = [
            (self.corpus[i].doc_id, float(similarities[i]), self.corpus[i].text)
            for i in range(len(self.corpus))
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score, text) in enumerate(scored_docs[:top_k], start=1):
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                text=text
            ))
        
        return results


class ScoreFusion:
    """Score fusion methods for combining retrieval results."""
    
    @staticmethod
    def normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Min-max normalize scores to [0, 1] range.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of results with normalized scores
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [
                RetrievalResult(
                    doc_id=r.doc_id,
                    score=1.0,
                    rank=r.rank,
                    text=r.text
                )
                for r in results
            ]
        
        return [
            RetrievalResult(
                doc_id=r.doc_id,
                score=(r.score - min_score) / (max_score - min_score),
                rank=r.rank,
                text=r.text
            )
            for r in results
        ]
    
    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[List[RetrievalResult]],
        k: int = 60,
        top_k: int = 100
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score for document d: sum(1 / (k + rank(d))) across all result lists
        
        Args:
            results_list: List of result lists from different retrievers
            k: RRF constant (default 60)
            top_k: Number of results to return
            
        Returns:
            Combined and re-ranked results
        """
        doc_scores: Dict[str, float] = {}
        doc_texts: Dict[str, str] = {}
        
        for results in results_list:
            for result in results:
                rrf_score = 1.0 / (k + result.rank)
                doc_scores[result.doc_id] = doc_scores.get(result.doc_id, 0) + rrf_score
                if result.text:
                    doc_texts[result.doc_id] = result.text
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                text=doc_texts.get(doc_id)
            )
            for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1)
        ]
    
    @staticmethod
    def weighted_combination(
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        alpha: float = 0.5,
        top_k: int = 100,
        normalize: bool = True
    ) -> List[RetrievalResult]:
        """
        Combine results using weighted linear combination.
        
        Combined score: alpha * bm25_score + (1 - alpha) * dense_score
        
        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            alpha: Weight for BM25 scores (1 - alpha for dense)
            top_k: Number of results to return
            normalize: Whether to normalize scores before combination
            
        Returns:
            Combined and re-ranked results
        """
        if normalize:
            bm25_results = ScoreFusion.normalize_scores(bm25_results)
            dense_results = ScoreFusion.normalize_scores(dense_results)
        
        bm25_scores = {r.doc_id: r.score for r in bm25_results}
        dense_scores = {r.doc_id: r.score for r in dense_results}
        doc_texts = {r.doc_id: r.text for r in bm25_results if r.text}
        doc_texts.update({r.doc_id: r.text for r in dense_results if r.text})
        
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        combined_scores = {}
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0)
            dense_score = dense_scores.get(doc_id, 0)
            combined_scores[doc_id] = alpha * bm25_score + (1 - alpha) * dense_score
        
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                text=doc_texts.get(doc_id)
            )
            for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1)
        ]


class EvaluationMetrics:
    """Evaluation metrics for retrieval systems."""
    
    @staticmethod
    def mean_reciprocal_rank(
        results_list: List[List[RetrievalResult]],
        relevance: Dict[str, List[str]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results_list: List of query results (each is a list of RetrievalResult)
            relevance: Dict mapping query_id to list of relevant doc_ids
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for query_idx, results in enumerate(results_list):
            query_id = str(query_idx)
            relevant_docs = set(relevance.get(query_id, []))
            
            rr = 0.0
            for result in results:
                if result.doc_id in relevant_docs:
                    rr = 1.0 / result.rank
                    break
            reciprocal_ranks.append(rr)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(
        results_list: List[List[RetrievalResult]],
        relevance: Dict[str, Dict[str, int]],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k (NDCG@k).
        
        Args:
            results_list: List of query results
            relevance: Dict mapping query_id to dict of doc_id -> relevance_score
            k: Cutoff for evaluation
            
        Returns:
            NDCG@k score
        """
        ndcg_scores = []
        
        for query_idx, results in enumerate(results_list):
            query_id = str(query_idx)
            rel_scores = relevance.get(query_id, {})
            
            dcg = 0.0
            for i, result in enumerate(results[:k]):
                rel = rel_scores.get(result.doc_id, 0)
                dcg += (2 ** rel - 1) / math.log2(i + 2)
            
            ideal_rels = sorted(rel_scores.values(), reverse=True)[:k]
            idcg = sum(
                (2 ** rel - 1) / math.log2(i + 2)
                for i, rel in enumerate(ideal_rels)
            )
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    @staticmethod
    def recall_at_k(
        results_list: List[List[RetrievalResult]],
        relevance: Dict[str, List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate Recall at k.
        
        Args:
            results_list: List of query results
            relevance: Dict mapping query_id to list of relevant doc_ids
            k: Cutoff for evaluation
            
        Returns:
            Recall@k score
        """
        recall_scores = []
        
        for query_idx, results in enumerate(results_list):
            query_id = str(query_idx)
            relevant_docs = set(relevance.get(query_id, []))
            
            if not relevant_docs:
                continue
            
            retrieved_at_k = {r.doc_id for r in results[:k]}
            hits = len(retrieved_at_k & relevant_docs)
            recall = hits / len(relevant_docs)
            recall_scores.append(recall)
        
        return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0


class HybridRetriever:
    """
    Hybrid retrieval pipeline combining BM25 and dense retrieval.
    
    Supports multiple fusion strategies and parameter tuning.
    """
    
    def __init__(
        self,
        bm25_params: Optional[Dict[str, float]] = None,
        dense_model: str = "all-MiniLM-L6-v2",
        fusion_method: str = "rrf",
        fusion_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_params: Parameters for BM25 (k1, b, epsilon)
            dense_model: Name of sentence-transformers model
            fusion_method: Fusion method ('rrf' or 'weighted')
            fusion_params: Parameters for fusion method
        """
        bm25_params = bm25_params or {}
        self.bm25 = BM25Retriever(**bm25_params)
        self.dense = DenseRetriever(model_name=dense_model)
        self.fusion_method = fusion_method
        self.fusion_params = fusion_params or {}
        self._indexed = False
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents for both retrievers.
        
        Args:
            documents: List of Document objects to index
        """
        self.bm25.index(documents)
        self.dense.index(documents)
        self._indexed = True
    
    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        return_individual: bool = False
    ) -> Union[List[RetrievalResult], Tuple[List[RetrievalResult], List[RetrievalResult], List[RetrievalResult]]]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            return_individual: If True, also return individual retriever results
            
        Returns:
            Hybrid results, or tuple of (hybrid, bm25, dense) if return_individual=True
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")
        
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        dense_results = self.dense.retrieve(query, top_k=top_k)
        
        if self.fusion_method == "rrf":
            k = self.fusion_params.get("k", 60)
            hybrid_results = ScoreFusion.reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=k,
                top_k=top_k
            )
        elif self.fusion_method == "weighted":
            alpha = self.fusion_params.get("alpha", 0.5)
            hybrid_results = ScoreFusion.weighted_combination(
                bm25_results,
                dense_results,
                alpha=alpha,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        if return_individual:
            return hybrid_results, bm25_results, dense_results
        return hybrid_results
    
    def tune_fusion_weights(
        self,
        queries: List[str],
        relevance: Dict[str, List[str]],
        metric: str = "mrr",
        alpha_range: Optional[List[float]] = None,
        rrf_k_range: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Tune fusion parameters using grid search.
        
        Args:
            queries: List of query strings
            relevance: Dict mapping query_id to list of relevant doc_ids
            metric: Metric to optimize ('mrr', 'recall@10', 'recall@100')
            alpha_range: Range of alpha values for weighted fusion
            rrf_k_range: Range of k values for RRF
            
        Returns:
            Dict with best parameters and scores
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")
        
        alpha_range = alpha_range or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rrf_k_range = rrf_k_range or [10, 30, 60, 100]
        
        all_bm25_results = []
        all_dense_results = []
        for query in queries:
            bm25_results = self.bm25.retrieve(query, top_k=100)
            dense_results = self.dense.retrieve(query, top_k=100)
            all_bm25_results.append(bm25_results)
            all_dense_results.append(dense_results)
        
        def evaluate_metric(results_list: List[List[RetrievalResult]]) -> float:
            if metric == "mrr":
                return EvaluationMetrics.mean_reciprocal_rank(results_list, relevance)
            elif metric == "recall@10":
                return EvaluationMetrics.recall_at_k(results_list, relevance, k=10)
            elif metric == "recall@100":
                return EvaluationMetrics.recall_at_k(results_list, relevance, k=100)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        results = {
            "weighted": [],
            "rrf": [],
            "best_weighted": {"alpha": 0.5, "score": 0.0},
            "best_rrf": {"k": 60, "score": 0.0}
        }
        
        for alpha in alpha_range:
            hybrid_results = [
                ScoreFusion.weighted_combination(
                    bm25_results, dense_results, alpha=alpha, top_k=100
                )
                for bm25_results, dense_results in zip(all_bm25_results, all_dense_results)
            ]
            score = evaluate_metric(hybrid_results)
            results["weighted"].append({"alpha": alpha, "score": score})
            if score > results["best_weighted"]["score"]:
                results["best_weighted"] = {"alpha": alpha, "score": score}
        
        for k in rrf_k_range:
            hybrid_results = [
                ScoreFusion.reciprocal_rank_fusion(
                    [bm25_results, dense_results], k=k, top_k=100
                )
                for bm25_results, dense_results in zip(all_bm25_results, all_dense_results)
            ]
            score = evaluate_metric(hybrid_results)
            results["rrf"].append({"k": k, "score": score})
            if score > results["best_rrf"]["score"]:
                results["best_rrf"] = {"k": k, "score": score}
        
        return results


def run_evaluation(
    hybrid_retriever: HybridRetriever,
    queries: List[str],
    relevance_binary: Dict[str, List[str]],
    relevance_graded: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    """
    Run full evaluation of hybrid retrieval pipeline.
    
    Args:
        hybrid_retriever: Configured HybridRetriever instance
        queries: List of query strings
        relevance_binary: Binary relevance judgments (query_id -> [doc_ids])
        relevance_graded: Graded relevance judgments (query_id -> {doc_id: score})
        
    Returns:
        Dict with evaluation results for all methods
    """
    bm25_results_all = []
    dense_results_all = []
    hybrid_results_all = []
    
    for query in queries:
        hybrid, bm25, dense = hybrid_retriever.retrieve(
            query, top_k=100, return_individual=True
        )
        bm25_results_all.append(bm25)
        dense_results_all.append(dense)
        hybrid_results_all.append(hybrid)
    
    results = {
        "bm25": {
            "mrr": EvaluationMetrics.mean_reciprocal_rank(bm25_results_all, relevance_binary),
            "ndcg@10": EvaluationMetrics.ndcg_at_k(bm25_results_all, relevance_graded, k=10),
            "recall@10": EvaluationMetrics.recall_at_k(bm25_results_all, relevance_binary, k=10),
            "recall@100": EvaluationMetrics.recall_at_k(bm25_results_all, relevance_binary, k=100)
        },
        "dense": {
            "mrr": EvaluationMetrics.mean_reciprocal_rank(dense_results_all, relevance_binary),
            "ndcg@10": EvaluationMetrics.ndcg_at_k(dense_results_all, relevance_graded, k=10),
            "recall@10": EvaluationMetrics.recall_at_k(dense_results_all, relevance_binary, k=10),
            "recall@100": EvaluationMetrics.recall_at_k(dense_results_all, relevance_binary, k=100)
        },
        "hybrid": {
            "mrr": EvaluationMetrics.mean_reciprocal_rank(hybrid_results_all, relevance_binary),
            "ndcg@10": EvaluationMetrics.ndcg_at_k(hybrid_results_all, relevance_graded, k=10),
            "recall@10": EvaluationMetrics.recall_at_k(hybrid_results_all, relevance_binary, k=10),
            "recall@100": EvaluationMetrics.recall_at_k(hybrid_results_all, relevance_binary, k=100)
        }
    }
    
    return results


def create_sample_dataset() -> Tuple[List[Document], List[str], Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """
    Create a sample dataset for testing and demonstration.
    
    Returns:
        Tuple of (documents, queries, binary_relevance, graded_relevance)
    """
    documents = [
        Document("doc_0", "Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
        Document("doc_1", "Deep learning uses neural networks with many layers to model complex patterns in data."),
        Document("doc_2", "Natural language processing helps computers understand and generate human language."),
        Document("doc_3", "Computer vision enables machines to interpret and understand visual information from the world."),
        Document("doc_4", "Reinforcement learning trains agents to make decisions by rewarding desired behaviors."),
        Document("doc_5", "Transfer learning allows models trained on one task to be applied to different but related tasks."),
        Document("doc_6", "Supervised learning uses labeled data to train models for classification and regression tasks."),
        Document("doc_7", "Unsupervised learning finds hidden patterns in data without labeled examples."),
        Document("doc_8", "Neural networks are computing systems inspired by biological neural networks in the brain."),
        Document("doc_9", "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning."),
        Document("doc_10", "Convolutional neural networks are specialized for processing grid-like data such as images."),
        Document("doc_11", "Recurrent neural networks are designed to handle sequential data like text and time series."),
        Document("doc_12", "Transformers use attention mechanisms to process sequences in parallel, revolutionizing NLP."),
        Document("doc_13", "BERT is a pre-trained language model that understands context from both directions."),
        Document("doc_14", "GPT models are autoregressive language models trained to predict the next token in a sequence."),
        Document("doc_15", "Word embeddings represent words as dense vectors capturing semantic relationships."),
        Document("doc_16", "Attention mechanisms allow models to focus on relevant parts of the input when making predictions."),
        Document("doc_17", "Batch normalization helps stabilize and accelerate training of deep neural networks."),
        Document("doc_18", "Dropout is a regularization technique that randomly sets neurons to zero during training."),
        Document("doc_19", "Cross-entropy loss is commonly used for classification tasks in machine learning."),
    ]
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing",
        "What are transformers in deep learning?",
        "How does reinforcement learning train agents?",
    ]
    
    binary_relevance = {
        "0": ["doc_0", "doc_6", "doc_7", "doc_9"],
        "1": ["doc_1", "doc_8", "doc_10", "doc_11", "doc_17", "doc_18"],
        "2": ["doc_2", "doc_12", "doc_13", "doc_14", "doc_15"],
        "3": ["doc_12", "doc_16", "doc_13", "doc_14"],
        "4": ["doc_4"],
    }
    
    graded_relevance = {
        "0": {"doc_0": 3, "doc_6": 2, "doc_7": 2, "doc_9": 1, "doc_1": 1},
        "1": {"doc_8": 3, "doc_1": 2, "doc_10": 2, "doc_11": 2, "doc_17": 1, "doc_18": 1},
        "2": {"doc_2": 3, "doc_12": 2, "doc_13": 2, "doc_14": 2, "doc_15": 1},
        "3": {"doc_12": 3, "doc_16": 2, "doc_13": 2, "doc_14": 1},
        "4": {"doc_4": 3},
    }
    
    return documents, queries, binary_relevance, graded_relevance


def main():
    """Run hybrid retrieval evaluation and save results."""
    print("Creating sample dataset...")
    documents, queries, binary_relevance, graded_relevance = create_sample_dataset()
    
    print("Initializing hybrid retriever with RRF fusion...")
    hybrid_rrf = HybridRetriever(
        fusion_method="rrf",
        fusion_params={"k": 60}
    )
    hybrid_rrf.index(documents)
    
    print("Running evaluation with RRF fusion...")
    rrf_results = run_evaluation(hybrid_rrf, queries, binary_relevance, graded_relevance)
    
    print("Initializing hybrid retriever with weighted fusion...")
    hybrid_weighted = HybridRetriever(
        fusion_method="weighted",
        fusion_params={"alpha": 0.5}
    )
    hybrid_weighted.index(documents)
    
    print("Running evaluation with weighted fusion...")
    weighted_results = run_evaluation(hybrid_weighted, queries, binary_relevance, graded_relevance)
    
    print("Tuning fusion weights...")
    tuning_results = hybrid_rrf.tune_fusion_weights(
        queries, binary_relevance, metric="mrr"
    )
    
    best_alpha = tuning_results["best_weighted"]["alpha"]
    hybrid_tuned = HybridRetriever(
        fusion_method="weighted",
        fusion_params={"alpha": best_alpha}
    )
    hybrid_tuned.index(documents)
    tuned_results = run_evaluation(hybrid_tuned, queries, binary_relevance, graded_relevance)
    
    final_results = {
        "rrf_fusion": {
            "parameters": {"k": 60},
            "metrics": rrf_results
        },
        "weighted_fusion": {
            "parameters": {"alpha": 0.5},
            "metrics": weighted_results
        },
        "tuned_weighted_fusion": {
            "parameters": {"alpha": best_alpha},
            "metrics": tuned_results
        },
        "tuning_results": tuning_results,
        "analysis": {
            "description": "Comparison of hybrid retrieval methods against individual baselines",
            "findings": []
        }
    }
    
    bm25_mrr = rrf_results["bm25"]["mrr"]
    dense_mrr = rrf_results["dense"]["mrr"]
    hybrid_rrf_mrr = rrf_results["hybrid"]["mrr"]
    hybrid_weighted_mrr = weighted_results["hybrid"]["mrr"]
    hybrid_tuned_mrr = tuned_results["hybrid"]["mrr"]
    
    findings = []
    if hybrid_rrf_mrr > bm25_mrr:
        findings.append(f"RRF hybrid improves MRR by {(hybrid_rrf_mrr - bm25_mrr) * 100:.1f}% over BM25")
    if hybrid_rrf_mrr > dense_mrr:
        findings.append(f"RRF hybrid improves MRR by {(hybrid_rrf_mrr - dense_mrr) * 100:.1f}% over dense retrieval")
    if hybrid_tuned_mrr > hybrid_weighted_mrr:
        findings.append(f"Tuned weighted fusion (alpha={best_alpha}) improves MRR by {(hybrid_tuned_mrr - hybrid_weighted_mrr) * 100:.1f}% over default alpha=0.5")
    
    best_method = max([
        ("BM25", bm25_mrr),
        ("Dense", dense_mrr),
        ("Hybrid RRF", hybrid_rrf_mrr),
        ("Hybrid Weighted", hybrid_weighted_mrr),
        ("Hybrid Tuned", hybrid_tuned_mrr)
    ], key=lambda x: x[1])
    findings.append(f"Best performing method: {best_method[0]} with MRR={best_method[1]:.4f}")
    
    final_results["analysis"]["findings"] = findings
    
    output_path = "hybrid_results.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\n=== Evaluation Summary ===")
    print(f"BM25 MRR: {bm25_mrr:.4f}")
    print(f"Dense MRR: {dense_mrr:.4f}")
    print(f"Hybrid RRF MRR: {hybrid_rrf_mrr:.4f}")
    print(f"Hybrid Weighted MRR: {hybrid_weighted_mrr:.4f}")
    print(f"Hybrid Tuned MRR: {hybrid_tuned_mrr:.4f}")
    print(f"\nBest alpha for weighted fusion: {best_alpha}")
    print(f"Best k for RRF: {tuning_results['best_rrf']['k']}")
    
    return final_results


if __name__ == "__main__":
    main()
