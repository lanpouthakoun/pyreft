"""
Retrieval Evaluation Pipeline

Evaluates and compares different retrieval methods:
- BM25 only
- Dense only
- Hybrid with various fusion strategies

Generates a scoreboard showing comparative performance.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .fusion import RRFFusion, WeightedFusion, LearnedFusion
from .metrics import RetrievalMetrics
from .synthetic_data import SyntheticDataset, SyntheticDataGenerator


@dataclass
class EvaluationResult:
    """Results from evaluating a retrieval method."""
    method_name: str
    metrics: Dict[str, float]
    per_query_metrics: List[Dict[str, float]]


@dataclass
class Scoreboard:
    """Comparative scoreboard of retrieval methods."""
    results: List[EvaluationResult]
    best_method_per_metric: Dict[str, str]
    hybrid_improvements: Dict[str, Dict[str, float]]
    recommendations: List[str]


class RetrievalEvaluator:
    """
    Evaluator for comparing retrieval methods.

    Runs comprehensive evaluation and generates scoreboard.
    """

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        k_values: List[int] = [1, 3, 5, 10],
    ):
        """
        Initialize the evaluator.

        Args:
            dense_model_name: Sentence transformer model name
            k_values: K values for metrics computation
        """
        self.dense_model_name = dense_model_name
        self.k_values = k_values
        self.hybrid_retriever: Optional[HybridRetriever] = None

    def _setup_retriever(self, documents: List[str]) -> None:
        """Set up the hybrid retriever with documents."""
        self.hybrid_retriever = HybridRetriever(
            dense_model_name=self.dense_model_name,
            fusion_strategy="rrf",
            documents=documents,
        )

    def _evaluate_method(
        self,
        method_name: str,
        retrieve_fn,
        queries: List[str],
        relevance_labels: List[Dict[int, int]],
        top_k: int = 10,
    ) -> EvaluationResult:
        """
        Evaluate a single retrieval method.

        Args:
            method_name: Name of the method
            retrieve_fn: Function that takes query and returns results
            queries: List of queries
            relevance_labels: Relevance labels for each query
            top_k: Number of results to retrieve

        Returns:
            EvaluationResult with metrics
        """
        per_query_metrics = []

        for query, relevance in zip(queries, relevance_labels):
            results = retrieve_fn(query, top_k)
            retrieved = [doc_idx for doc_idx, _, _ in results]
            relevant = set(relevance.keys())

            metrics = RetrievalMetrics.compute_all_metrics(
                retrieved=retrieved,
                relevant=relevant,
                relevance_scores=relevance,
                k_values=self.k_values,
            )
            per_query_metrics.append(metrics)

        aggregated = RetrievalMetrics.aggregate_metrics(per_query_metrics)

        return EvaluationResult(
            method_name=method_name,
            metrics=aggregated,
            per_query_metrics=per_query_metrics,
        )

    def evaluate_all_methods(
        self,
        documents: List[str],
        queries: List[str],
        relevance_labels: List[Dict[int, int]],
        top_k: int = 10,
        alpha_values: List[float] = [0.3, 0.5, 0.7],
    ) -> List[EvaluationResult]:
        """
        Evaluate all retrieval methods.

        Args:
            documents: List of documents
            queries: List of queries
            relevance_labels: Relevance labels for each query
            top_k: Number of results to retrieve
            alpha_values: Alpha values for weighted fusion

        Returns:
            List of EvaluationResult for each method
        """
        self._setup_retriever(documents)
        if self.hybrid_retriever is None:
            raise ValueError("Failed to set up retriever")

        results = []

        bm25_result = self._evaluate_method(
            method_name="BM25",
            retrieve_fn=self.hybrid_retriever.retrieve_bm25_only,
            queries=queries,
            relevance_labels=relevance_labels,
            top_k=top_k,
        )
        results.append(bm25_result)

        dense_result = self._evaluate_method(
            method_name="Dense",
            retrieve_fn=self.hybrid_retriever.retrieve_dense_only,
            queries=queries,
            relevance_labels=relevance_labels,
            top_k=top_k,
        )
        results.append(dense_result)

        self.hybrid_retriever.set_fusion_strategy("rrf")
        rrf_result = self._evaluate_method(
            method_name="Hybrid (RRF)",
            retrieve_fn=self.hybrid_retriever.retrieve,
            queries=queries,
            relevance_labels=relevance_labels,
            top_k=top_k,
        )
        results.append(rrf_result)

        for alpha in alpha_values:
            self.hybrid_retriever.set_fusion_strategy(
                "weighted", {"alpha": alpha}
            )
            weighted_result = self._evaluate_method(
                method_name=f"Hybrid (Weighted a={alpha})",
                retrieve_fn=self.hybrid_retriever.retrieve,
                queries=queries,
                relevance_labels=relevance_labels,
                top_k=top_k,
            )
            results.append(weighted_result)

        learned_fusion = LearnedFusion()

        bm25_results_list = []
        dense_results_list = []
        for query in queries:
            bm25_results_list.append(
                self.hybrid_retriever.retrieve_bm25_only(query, top_k * 2)
            )
            dense_results_list.append(
                self.hybrid_retriever.retrieve_dense_only(query, top_k * 2)
            )

        train_size = min(50, len(queries) // 2)
        if train_size > 0:
            learned_fusion.train(
                queries=queries[:train_size],
                relevance_labels=relevance_labels[:train_size],
                bm25_results_list=bm25_results_list[:train_size],
                dense_results_list=dense_results_list[:train_size],
            )

        self.hybrid_retriever.set_fusion_strategy(learned_fusion)
        learned_result = self._evaluate_method(
            method_name="Hybrid (Learned)",
            retrieve_fn=self.hybrid_retriever.retrieve,
            queries=queries,
            relevance_labels=relevance_labels,
            top_k=top_k,
        )
        results.append(learned_result)

        return results

    def generate_scoreboard(
        self,
        results: List[EvaluationResult],
    ) -> Scoreboard:
        """
        Generate a comparative scoreboard.

        Args:
            results: List of evaluation results

        Returns:
            Scoreboard with comparisons and recommendations
        """
        best_method_per_metric: Dict[str, str] = {}
        if results:
            metric_names = results[0].metrics.keys()
            for metric in metric_names:
                best_score = -float('inf')
                best_method = ""
                for result in results:
                    if result.metrics.get(metric, 0) > best_score:
                        best_score = result.metrics[metric]
                        best_method = result.method_name
                best_method_per_metric[metric] = best_method

        hybrid_improvements: Dict[str, Dict[str, float]] = {}
        bm25_metrics = None
        dense_metrics = None

        for result in results:
            if result.method_name == "BM25":
                bm25_metrics = result.metrics
            elif result.method_name == "Dense":
                dense_metrics = result.metrics

        if bm25_metrics and dense_metrics:
            for result in results:
                if "Hybrid" in result.method_name:
                    improvements = {}
                    for metric, value in result.metrics.items():
                        best_baseline = max(
                            bm25_metrics.get(metric, 0),
                            dense_metrics.get(metric, 0),
                        )
                        if best_baseline > 0:
                            improvement = (
                                (value - best_baseline) / best_baseline * 100
                            )
                            improvements[metric] = improvement
                    hybrid_improvements[result.method_name] = improvements

        recommendations = self._generate_recommendations(
            results, hybrid_improvements
        )

        return Scoreboard(
            results=results,
            best_method_per_metric=best_method_per_metric,
            hybrid_improvements=hybrid_improvements,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        results: List[EvaluationResult],
        hybrid_improvements: Dict[str, Dict[str, float]],
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        best_hybrid = None
        best_avg_improvement = -float('inf')

        for method_name, improvements in hybrid_improvements.items():
            if improvements:
                avg_improvement = np.mean(list(improvements.values()))
                if avg_improvement > best_avg_improvement:
                    best_avg_improvement = avg_improvement
                    best_hybrid = method_name

        if best_hybrid:
            recommendations.append(
                f"Best hybrid method: {best_hybrid} with average "
                f"improvement of {best_avg_improvement:.2f}% over baselines."
            )

        bm25_metrics = None
        dense_metrics = None
        for result in results:
            if result.method_name == "BM25":
                bm25_metrics = result.metrics
            elif result.method_name == "Dense":
                dense_metrics = result.metrics

        if bm25_metrics and dense_metrics:
            bm25_ndcg = bm25_metrics.get("ndcg@10", 0)
            dense_ndcg = dense_metrics.get("ndcg@10", 0)

            if bm25_ndcg > dense_ndcg * 1.1:
                recommendations.append(
                    "BM25 significantly outperforms dense retrieval. "
                    "Consider using higher BM25 weight (alpha > 0.6) in weighted fusion."
                )
            elif dense_ndcg > bm25_ndcg * 1.1:
                recommendations.append(
                    "Dense retrieval significantly outperforms BM25. "
                    "Consider using lower BM25 weight (alpha < 0.4) in weighted fusion."
                )
            else:
                recommendations.append(
                    "BM25 and dense retrieval perform similarly. "
                    "Use balanced weights (alpha = 0.5) or RRF fusion."
                )

        rrf_improvement = hybrid_improvements.get("Hybrid (RRF)", {})
        if rrf_improvement:
            avg_rrf = np.mean(list(rrf_improvement.values()))
            if avg_rrf > 5:
                recommendations.append(
                    "RRF fusion shows strong improvement. "
                    "It's a good default choice as it requires no tuning."
                )

        weighted_results = [
            (name, impr)
            for name, impr in hybrid_improvements.items()
            if "Weighted" in name
        ]
        if weighted_results:
            best_weighted = max(
                weighted_results,
                key=lambda x: np.mean(list(x[1].values())) if x[1] else 0,
            )
            if best_weighted[1]:
                recommendations.append(
                    f"Best weighted fusion: {best_weighted[0]}. "
                    "Fine-tune alpha around this value for optimal results."
                )

        learned_improvement = hybrid_improvements.get("Hybrid (Learned)", {})
        if learned_improvement:
            avg_learned = np.mean(list(learned_improvement.values()))
            if avg_learned > best_avg_improvement * 0.9:
                recommendations.append(
                    "Learned fusion performs well. Consider training on "
                    "more data for potentially better results."
                )

        recommendations.append(
            "For production use, validate on held-out data and consider "
            "query-type-specific fusion strategies."
        )

        return recommendations

    def print_scoreboard(self, scoreboard: Scoreboard) -> str:
        """
        Format scoreboard as a printable string.

        Args:
            scoreboard: Scoreboard to format

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 80)
        lines.append("HYBRID RETRIEVAL EVALUATION SCOREBOARD")
        lines.append("=" * 80)
        lines.append("")

        lines.append("METRICS BY METHOD")
        lines.append("-" * 80)

        header = f"{'Method':<30}"
        key_metrics = ["ndcg@10", "mrr", "precision@5", "recall@10"]
        for metric in key_metrics:
            header += f"{metric:>12}"
        lines.append(header)
        lines.append("-" * 80)

        for result in scoreboard.results:
            row = f"{result.method_name:<30}"
            for metric in key_metrics:
                value = result.metrics.get(metric, 0)
                row += f"{value:>12.4f}"
            lines.append(row)

        lines.append("")
        lines.append("BEST METHOD PER METRIC")
        lines.append("-" * 80)

        for metric in key_metrics:
            best = scoreboard.best_method_per_metric.get(metric, "N/A")
            lines.append(f"  {metric:<20}: {best}")

        lines.append("")
        lines.append("HYBRID IMPROVEMENTS OVER BASELINES (%)")
        lines.append("-" * 80)

        for method_name, improvements in scoreboard.hybrid_improvements.items():
            lines.append(f"\n  {method_name}:")
            for metric in key_metrics:
                if metric in improvements:
                    imp = improvements[metric]
                    sign = "+" if imp > 0 else ""
                    lines.append(f"    {metric:<20}: {sign}{imp:.2f}%")

        lines.append("")
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)

        for i, rec in enumerate(scoreboard.recommendations, 1):
            lines.append(f"  {i}. {rec}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def run_full_evaluation(
        self,
        num_documents: int = 200,
        num_queries: int = 100,
        seed: int = 42,
    ) -> Tuple[Scoreboard, str]:
        """
        Run a full evaluation with synthetic data.

        Args:
            num_documents: Number of documents to generate
            num_queries: Number of queries to generate
            seed: Random seed for reproducibility

        Returns:
            Tuple of (Scoreboard, formatted_output)
        """
        generator = SyntheticDataGenerator(seed=seed)
        dataset = generator.generate(
            num_documents=num_documents,
            num_queries=num_queries,
        )

        results = self.evaluate_all_methods(
            documents=dataset.documents,
            queries=dataset.queries,
            relevance_labels=dataset.relevance_labels,
        )

        scoreboard = self.generate_scoreboard(results)
        output = self.print_scoreboard(scoreboard)

        return scoreboard, output


def run_evaluation_demo() -> str:
    """
    Run a demonstration evaluation.

    Returns:
        Formatted scoreboard output
    """
    evaluator = RetrievalEvaluator()
    _, output = evaluator.run_full_evaluation(
        num_documents=200,
        num_queries=100,
        seed=42,
    )
    return output
