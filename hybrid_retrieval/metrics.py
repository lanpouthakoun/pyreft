"""
Retrieval Evaluation Metrics

Implements standard information retrieval metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
"""

from typing import List, Dict, Set, Optional
import numpy as np


class RetrievalMetrics:
    """
    Calculator for retrieval evaluation metrics.

    All metrics assume higher scores indicate better performance.
    """

    @staticmethod
    def precision_at_k(
        retrieved: List[int],
        relevant: Set[int],
        k: int,
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K = (# of relevant docs in top K) / K

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevant: Set of relevant document indices
            k: Number of top results to consider

        Returns:
            Precision@K score between 0 and 1
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(
        retrieved: List[int],
        relevant: Set[int],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# of relevant docs in top K) / (total # of relevant docs)

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevant: Set of relevant document indices
            k: Number of top results to consider

        Returns:
            Recall@K score between 0 and 1
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        return relevant_in_top_k / len(relevant)

    @staticmethod
    def reciprocal_rank(
        retrieved: List[int],
        relevant: Set[int],
    ) -> float:
        """
        Calculate Reciprocal Rank.

        RR = 1 / (rank of first relevant document)

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevant: Set of relevant document indices

        Returns:
            Reciprocal rank score between 0 and 1
        """
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_lists: List[List[int]],
        relevant_sets: List[Set[int]],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = (1/|Q|) * sum(1/rank_i) for all queries

        Args:
            retrieved_lists: List of retrieved document lists for each query
            relevant_sets: List of relevant document sets for each query

        Returns:
            MRR score between 0 and 1
        """
        if not retrieved_lists:
            return 0.0

        rr_sum = sum(
            RetrievalMetrics.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in zip(retrieved_lists, relevant_sets)
        )
        return rr_sum / len(retrieved_lists)

    @staticmethod
    def dcg_at_k(
        retrieved: List[int],
        relevance_scores: Dict[int, int],
        k: int,
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevance_scores: Dictionary mapping doc_idx to relevance score
            k: Number of top results to consider

        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0)
            dcg += rel / np.log2(i + 2)
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[int],
        relevance_scores: Dict[int, int],
        k: int,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K

        where IDCG@K is the ideal DCG (perfect ranking).

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevance_scores: Dictionary mapping doc_idx to relevance score
            k: Number of top results to consider

        Returns:
            NDCG@K score between 0 and 1
        """
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, k)

        ideal_ranking = sorted(
            relevance_scores.values(), reverse=True
        )[:k]
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_ranking)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def average_precision(
        retrieved: List[int],
        relevant: Set[int],
    ) -> float:
        """
        Calculate Average Precision.

        AP = (1/|R|) * sum(P@k * rel(k)) for k in 1..n

        where rel(k) is 1 if doc at rank k is relevant, 0 otherwise.

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevant: Set of relevant document indices

        Returns:
            Average precision score between 0 and 1
        """
        if not relevant:
            return 0.0

        ap_sum = 0.0
        relevant_count = 0

        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                ap_sum += precision_at_rank

        return ap_sum / len(relevant)

    @staticmethod
    def mean_average_precision(
        retrieved_lists: List[List[int]],
        relevant_sets: List[Set[int]],
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).

        MAP = (1/|Q|) * sum(AP_q) for all queries

        Args:
            retrieved_lists: List of retrieved document lists for each query
            relevant_sets: List of relevant document sets for each query

        Returns:
            MAP score between 0 and 1
        """
        if not retrieved_lists:
            return 0.0

        ap_sum = sum(
            RetrievalMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in zip(retrieved_lists, relevant_sets)
        )
        return ap_sum / len(retrieved_lists)

    @staticmethod
    def f1_at_k(
        retrieved: List[int],
        relevant: Set[int],
        k: int,
    ) -> float:
        """
        Calculate F1 score at K.

        F1@K = 2 * (P@K * R@K) / (P@K + R@K)

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevant: Set of relevant document indices
            k: Number of top results to consider

        Returns:
            F1@K score between 0 and 1
        """
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def compute_all_metrics(
        retrieved: List[int],
        relevant: Set[int],
        relevance_scores: Optional[Dict[int, int]] = None,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single query.

        Args:
            retrieved: List of retrieved document indices (ranked)
            relevant: Set of relevant document indices
            relevance_scores: Optional graded relevance scores
            k_values: List of K values to compute metrics for

        Returns:
            Dictionary of metric names to scores
        """
        if relevance_scores is None:
            relevance_scores = {doc: 1 for doc in relevant}

        metrics = {}

        for k in k_values:
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(
                retrieved, relevant, k
            )
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(
                retrieved, relevant, k
            )
            metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(
                retrieved, relevance_scores, k
            )
            metrics[f"f1@{k}"] = RetrievalMetrics.f1_at_k(
                retrieved, relevant, k
            )

        metrics["mrr"] = RetrievalMetrics.reciprocal_rank(retrieved, relevant)
        metrics["map"] = RetrievalMetrics.average_precision(retrieved, relevant)

        return metrics

    @staticmethod
    def aggregate_metrics(
        all_metrics: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple queries.

        Args:
            all_metrics: List of metric dictionaries for each query

        Returns:
            Dictionary of averaged metrics
        """
        if not all_metrics:
            return {}

        aggregated = {}
        metric_names = all_metrics[0].keys()

        for name in metric_names:
            values = [m[name] for m in all_metrics if name in m]
            aggregated[name] = np.mean(values) if values else 0.0

        return aggregated
