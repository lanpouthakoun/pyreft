"""Scoreboard output module for retrieval evaluation results."""

import json
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class Scoreboard:
    """Scoreboard for displaying and exporting retrieval evaluation results."""

    def __init__(self, experiment_name: Optional[str] = None):
        """Initialize the scoreboard.

        Args:
            experiment_name: Name for this evaluation experiment.
        """
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.per_query_results: List[Dict] = []
        self.aggregate_results: Dict = {}
        self.model_info: Dict = {}
        self.dataset_info: Dict = {}
        self.recommendations: List[str] = []

    def add_per_query_results(self, results: List[Dict]) -> None:
        """Add per-query evaluation results.

        Args:
            results: List of dictionaries with per-query metrics.
        """
        self.per_query_results = results

    def add_aggregate_results(self, results: Dict) -> None:
        """Add aggregate evaluation results.

        Args:
            results: Dictionary with aggregate metrics.
        """
        self.aggregate_results = results

    def add_model_info(self, info: Dict) -> None:
        """Add model information.

        Args:
            info: Dictionary with model details.
        """
        self.model_info = info

    def add_dataset_info(self, info: Dict) -> None:
        """Add dataset information.

        Args:
            info: Dictionary with dataset details.
        """
        self.dataset_info = info

    def add_recommendations(self, recommendations: List[str]) -> None:
        """Add recommendations based on results.

        Args:
            recommendations: List of recommendation strings.
        """
        self.recommendations = recommendations

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the evaluation.

        Returns:
            Dictionary containing the full evaluation summary.
        """
        return {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_info": self.model_info,
            "dataset_info": self.dataset_info,
            "aggregate_metrics": self.aggregate_results,
            "recommendations": self.recommendations,
            "num_queries_evaluated": len(self.per_query_results)
        }

    def to_json(self, output_path: Optional[str] = None, include_per_query: bool = True) -> str:
        """Export results to JSON format.

        Args:
            output_path: Optional path to save the JSON file.
            include_per_query: Whether to include per-query results.

        Returns:
            JSON string of the results.
        """
        data = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_info": self.model_info,
            "dataset_info": self.dataset_info,
            "aggregate_metrics": self.aggregate_results,
            "recommendations": self.recommendations
        }

        if include_per_query:
            data["per_query_results"] = self.per_query_results

        json_str = json.dumps(data, indent=2)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_str)

        return json_str

    def to_csv(self, output_path: str) -> None:
        """Export per-query results to CSV format.

        Args:
            output_path: Path to save the CSV file.
        """
        if not self.per_query_results:
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(self.per_query_results[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.per_query_results)

    def to_aggregate_csv(self, output_path: str) -> None:
        """Export aggregate results to CSV format.

        Args:
            output_path: Path to save the CSV file.
        """
        if not self.aggregate_results:
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        rows = [{"metric": k, "value": v} for k, v in self.aggregate_results.items()]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)

    def print_summary(self) -> str:
        """Print a formatted summary of the evaluation.

        Returns:
            Formatted string summary.
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"RETRIEVAL EVALUATION SCOREBOARD")
        lines.append(f"Experiment: {self.experiment_name}")
        lines.append("=" * 60)

        if self.model_info:
            lines.append("\nModel Information:")
            lines.append(f"  Model: {self.model_info.get('model_name', 'N/A')}")
            lines.append(f"  Dimension: {self.model_info.get('dimension', 'N/A')}")
            lines.append(f"  Index Type: {self.model_info.get('index_type', 'N/A')}")
            lines.append(f"  Documents Indexed: {self.model_info.get('num_documents', 'N/A')}")

        if self.dataset_info:
            lines.append("\nDataset Information:")
            for key, value in self.dataset_info.items():
                lines.append(f"  {key}: {value}")

        if self.aggregate_results:
            lines.append("\nAggregate Metrics:")
            lines.append("-" * 40)

            mean_metrics = {k: v for k, v in self.aggregate_results.items() if k.startswith("mean_")}
            for metric, value in sorted(mean_metrics.items()):
                display_name = metric.replace("mean_", "").upper()
                lines.append(f"  {display_name}: {value:.4f}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            lines.append("-" * 40)
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.append("\n" + "=" * 60)

        summary = "\n".join(lines)
        print(summary)
        return summary


def generate_recommendations(
    aggregate_metrics: Dict,
    model_name: str,
    dataset_size: int
) -> List[str]:
    """Generate recommendations based on evaluation results.

    Args:
        aggregate_metrics: Dictionary of aggregate metrics.
        model_name: Name of the embedding model used.
        dataset_size: Number of documents in the dataset.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    mrr = aggregate_metrics.get("mean_mrr", 0)
    precision_1 = aggregate_metrics.get("mean_precision@1", 0)
    recall_10 = aggregate_metrics.get("mean_recall@10", 0)
    ndcg_10 = aggregate_metrics.get("mean_ndcg@10", 0)

    if mrr > 0.8:
        recommendations.append(
            f"Excellent MRR ({mrr:.3f}): The model consistently ranks relevant documents first. "
            "This configuration is well-suited for single-answer retrieval tasks."
        )
    elif mrr > 0.5:
        recommendations.append(
            f"Good MRR ({mrr:.3f}): Relevant documents typically appear in top positions. "
            "Consider fine-tuning or using a larger model for critical applications."
        )
    else:
        recommendations.append(
            f"Low MRR ({mrr:.3f}): Consider using a more powerful model like 'all-mpnet-base-v2' "
            "or fine-tuning on domain-specific data."
        )

    if precision_1 > 0.7:
        recommendations.append(
            f"High Precision@1 ({precision_1:.3f}): Excellent for applications requiring "
            "single best-match retrieval like FAQ systems."
        )

    if recall_10 > 0.8:
        recommendations.append(
            f"Strong Recall@10 ({recall_10:.3f}): Good coverage of relevant documents. "
            "Suitable for applications where finding all relevant items matters."
        )
    elif recall_10 < 0.5:
        recommendations.append(
            f"Low Recall@10 ({recall_10:.3f}): Consider increasing k or using hybrid retrieval "
            "(combining dense and sparse methods) for better coverage."
        )

    if "MiniLM" in model_name:
        recommendations.append(
            "Model Selection: MiniLM models offer excellent speed-quality tradeoff. "
            "For higher quality, consider 'all-mpnet-base-v2'. "
            "For multilingual support, use 'paraphrase-multilingual-MiniLM-L12-v2'."
        )

    if dataset_size < 1000:
        recommendations.append(
            "Index Type: For small datasets (<1000 docs), flat index provides best accuracy. "
            "Consider IVF or HNSW indexes for larger collections."
        )
    elif dataset_size < 100000:
        recommendations.append(
            "Index Type: For medium datasets, IVF index with nlist=100-1000 balances "
            "speed and accuracy well."
        )
    else:
        recommendations.append(
            "Index Type: For large datasets (>100k docs), HNSW index provides best "
            "query latency with minimal accuracy loss."
        )

    recommendations.append(
        "Use Cases: Dense retrieval excels at semantic similarity, paraphrase detection, "
        "and finding conceptually related content. For exact keyword matching, "
        "consider hybrid approaches combining dense and BM25 retrieval."
    )

    return recommendations
