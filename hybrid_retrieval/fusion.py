"""
Fusion Strategies for Hybrid Retrieval

Implements multiple methods for combining BM25 and dense retrieval scores:
- Reciprocal Rank Fusion (RRF)
- Weighted Combination
- Learned Fusion
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np


class BaseFusion(ABC):
    """Abstract base class for fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        bm25_results: List[Tuple[int, float, str]],
        dense_results: List[Tuple[int, float, str]],
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Fuse results from BM25 and dense retrieval.

        Args:
            bm25_results: Results from BM25 retriever (doc_idx, score, text)
            dense_results: Results from dense retriever (doc_idx, score, text)
            documents: Full list of documents
            top_k: Number of results to return

        Returns:
            Fused results as list of (doc_idx, score, text)
        """
        pass


class RRFFusion(BaseFusion):
    """
    Reciprocal Rank Fusion (RRF)

    Combines rankings using the formula:
    score(d) = sum(1 / (k + rank(d)))

    where k is a constant (typically 60) and rank(d) is the position
    of document d in each ranking.
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: Constant for RRF formula (default 60)
        """
        self.k = k

    def fuse(
        self,
        bm25_results: List[Tuple[int, float, str]],
        dense_results: List[Tuple[int, float, str]],
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Fuse results using Reciprocal Rank Fusion.

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            documents: Full list of documents
            top_k: Number of results to return

        Returns:
            Fused results
        """
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_idx, _, _) in enumerate(bm25_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (self.k + rank + 1)

        for rank, (doc_idx, _, _) in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (self.k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            results.append((doc_idx, score, documents[doc_idx]))

        return results


class WeightedFusion(BaseFusion):
    """
    Weighted Combination Fusion

    Combines normalized scores using linear weights:
    score(d) = alpha * norm_bm25(d) + (1 - alpha) * norm_dense(d)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize weighted fusion.

        Args:
            alpha: Weight for BM25 scores (1-alpha for dense)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Min-max normalize scores to [0, 1] range.

        Args:
            scores: Dictionary of doc_idx -> score

        Returns:
            Normalized scores
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }

    def fuse(
        self,
        bm25_results: List[Tuple[int, float, str]],
        dense_results: List[Tuple[int, float, str]],
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Fuse results using weighted combination.

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            documents: Full list of documents
            top_k: Number of results to return

        Returns:
            Fused results
        """
        bm25_scores = {doc_idx: score for doc_idx, score, _ in bm25_results}
        dense_scores = {doc_idx: score for doc_idx, score, _ in dense_results}

        norm_bm25 = self._normalize_scores(bm25_scores)
        norm_dense = self._normalize_scores(dense_scores)

        all_docs = set(norm_bm25.keys()) | set(norm_dense.keys())

        combined_scores: Dict[int, float] = {}
        for doc_idx in all_docs:
            bm25_score = norm_bm25.get(doc_idx, 0)
            dense_score = norm_dense.get(doc_idx, 0)
            combined_scores[doc_idx] = (
                self.alpha * bm25_score + (1 - self.alpha) * dense_score
            )

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            results.append((doc_idx, score, documents[doc_idx]))

        return results


class LearnedFusion(BaseFusion):
    """
    Learned Fusion

    Uses learned weights to combine scores. Weights can be trained
    on a validation set to optimize retrieval performance.
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        use_rank_features: bool = True,
    ):
        """
        Initialize learned fusion.

        Args:
            bm25_weight: Initial weight for BM25 scores
            dense_weight: Initial weight for dense scores
            use_rank_features: Whether to include rank-based features
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.use_rank_features = use_rank_features
        self._trained = False

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize scores to [0, 1] range."""
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }

    def _get_rank_score(self, rank: int, total: int) -> float:
        """Convert rank to a score (higher rank = higher score)."""
        if total <= 1:
            return 1.0
        return 1.0 - (rank / (total - 1))

    def train(
        self,
        queries: List[str],
        relevance_labels: List[Dict[int, int]],
        bm25_results_list: List[List[Tuple[int, float, str]]],
        dense_results_list: List[List[Tuple[int, float, str]]],
        learning_rate: float = 0.01,
        iterations: int = 100,
    ) -> None:
        """
        Train fusion weights using gradient descent on NDCG.

        Args:
            queries: List of training queries
            relevance_labels: List of relevance labels per query
            bm25_results_list: BM25 results for each query
            dense_results_list: Dense results for each query
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
        """
        best_bm25_weight = self.bm25_weight
        best_dense_weight = self.dense_weight
        best_score = -float('inf')

        for _ in range(iterations):
            for delta_bm25 in [-0.05, 0, 0.05]:
                for delta_dense in [-0.05, 0, 0.05]:
                    test_bm25 = max(0, min(1, self.bm25_weight + delta_bm25))
                    test_dense = max(0, min(1, self.dense_weight + delta_dense))

                    total_score = 0
                    for i, (bm25_res, dense_res) in enumerate(
                        zip(bm25_results_list, dense_results_list)
                    ):
                        if i >= len(relevance_labels):
                            continue

                        bm25_scores = {
                            doc_idx: score for doc_idx, score, _ in bm25_res
                        }
                        dense_scores = {
                            doc_idx: score for doc_idx, score, _ in dense_res
                        }

                        norm_bm25 = self._normalize_scores(bm25_scores)
                        norm_dense = self._normalize_scores(dense_scores)

                        all_docs = set(norm_bm25.keys()) | set(norm_dense.keys())
                        combined = {}
                        for doc_idx in all_docs:
                            combined[doc_idx] = (
                                test_bm25 * norm_bm25.get(doc_idx, 0)
                                + test_dense * norm_dense.get(doc_idx, 0)
                            )

                        sorted_docs = sorted(
                            combined.items(), key=lambda x: x[1], reverse=True
                        )
                        ranking = [doc_idx for doc_idx, _ in sorted_docs[:10]]

                        dcg = 0
                        for rank, doc_idx in enumerate(ranking):
                            rel = relevance_labels[i].get(doc_idx, 0)
                            dcg += rel / np.log2(rank + 2)

                        ideal_rels = sorted(
                            relevance_labels[i].values(), reverse=True
                        )[:10]
                        idcg = sum(
                            rel / np.log2(rank + 2)
                            for rank, rel in enumerate(ideal_rels)
                        )

                        if idcg > 0:
                            total_score += dcg / idcg

                    if total_score > best_score:
                        best_score = total_score
                        best_bm25_weight = test_bm25
                        best_dense_weight = test_dense

            self.bm25_weight = best_bm25_weight
            self.dense_weight = best_dense_weight

        self._trained = True

    def fuse(
        self,
        bm25_results: List[Tuple[int, float, str]],
        dense_results: List[Tuple[int, float, str]],
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float, str]]:
        """
        Fuse results using learned weights.

        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from dense retriever
            documents: Full list of documents
            top_k: Number of results to return

        Returns:
            Fused results
        """
        bm25_scores = {doc_idx: score for doc_idx, score, _ in bm25_results}
        dense_scores = {doc_idx: score for doc_idx, score, _ in dense_results}

        norm_bm25 = self._normalize_scores(bm25_scores)
        norm_dense = self._normalize_scores(dense_scores)

        all_docs = set(norm_bm25.keys()) | set(norm_dense.keys())

        combined_scores: Dict[int, float] = {}

        if self.use_rank_features:
            bm25_ranks = {
                doc_idx: rank for rank, (doc_idx, _, _) in enumerate(bm25_results)
            }
            dense_ranks = {
                doc_idx: rank for rank, (doc_idx, _, _) in enumerate(dense_results)
            }

            for doc_idx in all_docs:
                bm25_score = norm_bm25.get(doc_idx, 0)
                dense_score = norm_dense.get(doc_idx, 0)

                bm25_rank_score = self._get_rank_score(
                    bm25_ranks.get(doc_idx, len(bm25_results)),
                    len(bm25_results) + 1,
                )
                dense_rank_score = self._get_rank_score(
                    dense_ranks.get(doc_idx, len(dense_results)),
                    len(dense_results) + 1,
                )

                score_component = (
                    self.bm25_weight * bm25_score
                    + self.dense_weight * dense_score
                )
                rank_component = (
                    self.bm25_weight * bm25_rank_score
                    + self.dense_weight * dense_rank_score
                )

                combined_scores[doc_idx] = 0.7 * score_component + 0.3 * rank_component
        else:
            for doc_idx in all_docs:
                bm25_score = norm_bm25.get(doc_idx, 0)
                dense_score = norm_dense.get(doc_idx, 0)
                combined_scores[doc_idx] = (
                    self.bm25_weight * bm25_score
                    + self.dense_weight * dense_score
                )

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in sorted_docs[:top_k]:
            results.append((doc_idx, score, documents[doc_idx]))

        return results

    @property
    def is_trained(self) -> bool:
        """Check if the fusion model has been trained."""
        return self._trained

    def get_weights(self) -> Dict[str, float]:
        """Get current fusion weights."""
        return {
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
        }

    def set_weights(self, bm25_weight: float, dense_weight: float) -> None:
        """Set fusion weights manually."""
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
