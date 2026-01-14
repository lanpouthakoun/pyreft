"""Evaluation metrics for retrieval systems."""

import math
from typing import List, Dict, Set, Optional
import numpy as np


class RetrievalMetrics:
    """Evaluation metrics for information retrieval systems."""

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate Precision@K.

        Precision@K measures the proportion of retrieved documents in the top-k
        that are relevant.

        Args:
            retrieved: List of retrieved document IDs in ranked order.
            relevant: Set of relevant document IDs.
            k: Number of top results to consider.

        Returns:
            Precision@K score between 0 and 1.
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        if not top_k:
            return 0.0

        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
        return relevant_in_top_k / len(top_k)

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate Recall@K.

        Recall@K measures the proportion of relevant documents that appear
        in the top-k retrieved results.

        Args:
            retrieved: List of retrieved document IDs in ranked order.
            relevant: Set of relevant document IDs.
            k: Number of top results to consider.

        Returns:
            Recall@K score between 0 and 1.
        """
        if not relevant or k <= 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
        return relevant_in_top_k / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR) for a single query.

        MRR is the reciprocal of the rank of the first relevant document.

        Args:
            retrieved: List of retrieved document IDs in ranked order.
            relevant: Set of relevant document IDs.

        Returns:
            Reciprocal rank (1/rank of first relevant doc, or 0 if none found).
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def dcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """Calculate Discounted Cumulative Gain at K.

        DCG measures the usefulness of a document based on its position in the
        result list, with higher positions weighted more heavily.

        Args:
            retrieved: List of retrieved document IDs in ranked order.
            relevance_scores: Dictionary mapping document IDs to relevance scores.
            k: Number of top results to consider.

        Returns:
            DCG@K score.
        """
        if k <= 0:
            return 0.0

        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K.

        NDCG normalizes DCG by the ideal DCG (IDCG), which is the DCG of the
        perfect ranking.

        Args:
            retrieved: List of retrieved document IDs in ranked order.
            relevance_scores: Dictionary mapping document IDs to relevance scores.
            k: Number of top results to consider.

        Returns:
            NDCG@K score between 0 and 1.
        """
        if k <= 0:
            return 0.0

        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, k)

        ideal_order = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores[x],
            reverse=True
        )
        idcg = RetrievalMetrics.dcg_at_k(ideal_order, relevance_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def f1_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """Calculate F1 score at K.

        F1 is the harmonic mean of precision and recall.

        Args:
            retrieved: List of retrieved document IDs in ranked order.
            relevant: Set of relevant document IDs.
            k: Number of top results to consider.

        Returns:
            F1@K score between 0 and 1.
        """
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)


class RetrievalEvaluator:
    """Evaluator for retrieval systems across multiple queries."""

    def __init__(self, k_values: Optional[List[int]] = None):
        """Initialize the evaluator.

        Args:
            k_values: List of K values to evaluate at. Defaults to [1, 3, 5, 10].
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.metrics = RetrievalMetrics()
        self.results: List[Dict] = []

    def evaluate_query(
        self,
        query_id: str,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Evaluate retrieval results for a single query.

        Args:
            query_id: Identifier for the query.
            retrieved: List of retrieved document IDs in ranked order.
            relevant: Set of relevant document IDs.
            relevance_scores: Optional dictionary of graded relevance scores.

        Returns:
            Dictionary of metric scores for this query.
        """
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant}

        result = {"query_id": query_id}

        for k in self.k_values:
            result[f"precision@{k}"] = self.metrics.precision_at_k(retrieved, relevant, k)
            result[f"recall@{k}"] = self.metrics.recall_at_k(retrieved, relevant, k)
            result[f"f1@{k}"] = self.metrics.f1_at_k(retrieved, relevant, k)
            result[f"ndcg@{k}"] = self.metrics.ndcg_at_k(retrieved, relevance_scores, k)

        result["mrr"] = self.metrics.mean_reciprocal_rank(retrieved, relevant)

        self.results.append(result)
        return result

    def evaluate_batch(
        self,
        queries: List[Dict]
    ) -> List[Dict]:
        """Evaluate multiple queries.

        Args:
            queries: List of dictionaries with keys:
                - query_id: Query identifier
                - retrieved: List of retrieved document IDs
                - relevant: Set of relevant document IDs
                - relevance_scores: Optional graded relevance scores

        Returns:
            List of result dictionaries for each query.
        """
        results = []
        for query in queries:
            result = self.evaluate_query(
                query_id=query["query_id"],
                retrieved=query["retrieved"],
                relevant=query["relevant"],
                relevance_scores=query.get("relevance_scores")
            )
            results.append(result)
        return results

    def get_aggregate_metrics(self) -> Dict:
        """Calculate aggregate metrics across all evaluated queries.

        Returns:
            Dictionary of mean metric scores.
        """
        if not self.results:
            return {}

        aggregate = {}
        metric_keys = [k for k in self.results[0].keys() if k != "query_id"]

        for key in metric_keys:
            values = [r[key] for r in self.results]
            aggregate[f"mean_{key}"] = np.mean(values)
            aggregate[f"std_{key}"] = np.std(values)
            aggregate[f"min_{key}"] = np.min(values)
            aggregate[f"max_{key}"] = np.max(values)

        aggregate["num_queries"] = len(self.results)
        return aggregate

    def reset(self) -> None:
        """Reset the evaluator, clearing all stored results."""
        self.results = []
