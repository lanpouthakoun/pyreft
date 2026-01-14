"""
Unified Scoreboard Module

Aggregates metrics from all retrieval pipelines (BM25, Dense, Hybrid, ColBERT)
and provides a unified view of performance across different metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from pathlib import Path


class PipelineType(Enum):
    """Supported retrieval pipeline types."""
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    COLBERT = "colbert"


@dataclass
class PipelineMetrics:
    """
    Metrics for a single retrieval pipeline.
    
    Attributes:
        pipeline_type: Type of the retrieval pipeline
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        f1_score: F1 score (0-1)
        mrr: Mean Reciprocal Rank (0-1)
        ndcg: Normalized Discounted Cumulative Gain (0-1)
        ndcg_at_k: NDCG at different k values
        map_score: Mean Average Precision (0-1)
        latency_ms: Average latency in milliseconds
        latency_p50_ms: 50th percentile latency
        latency_p95_ms: 95th percentile latency
        latency_p99_ms: 99th percentile latency
        throughput_qps: Queries per second
        memory_mb: Memory usage in MB
        index_size_mb: Index size in MB
        cpu_utilization: CPU utilization percentage (0-100)
        gpu_utilization: GPU utilization percentage (0-100), if applicable
        num_queries: Number of queries evaluated
        metadata: Additional metadata
    """
    pipeline_type: PipelineType
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_qps: float = 0.0
    memory_mb: float = 0.0
    index_size_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: Optional[float] = None
    num_queries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "pipeline_type": self.pipeline_type.value,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "ndcg_at_k": self.ndcg_at_k,
            "map_score": self.map_score,
            "latency_ms": self.latency_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_qps": self.throughput_qps,
            "memory_mb": self.memory_mb,
            "index_size_mb": self.index_size_mb,
            "cpu_utilization": self.cpu_utilization,
            "gpu_utilization": self.gpu_utilization,
            "num_queries": self.num_queries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineMetrics":
        """Create PipelineMetrics from dictionary."""
        pipeline_type_str = data.get("pipeline_type", "bm25")
        pipeline_type = PipelineType(pipeline_type_str)
        
        return cls(
            pipeline_type=pipeline_type,
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            mrr=data.get("mrr", 0.0),
            ndcg=data.get("ndcg", 0.0),
            ndcg_at_k=data.get("ndcg_at_k", {}),
            map_score=data.get("map_score", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            latency_p50_ms=data.get("latency_p50_ms", 0.0),
            latency_p95_ms=data.get("latency_p95_ms", 0.0),
            latency_p99_ms=data.get("latency_p99_ms", 0.0),
            throughput_qps=data.get("throughput_qps", 0.0),
            memory_mb=data.get("memory_mb", 0.0),
            index_size_mb=data.get("index_size_mb", 0.0),
            cpu_utilization=data.get("cpu_utilization", 0.0),
            gpu_utilization=data.get("gpu_utilization"),
            num_queries=data.get("num_queries", 0),
            metadata=data.get("metadata", {}),
        )

    def get_accuracy_score(self) -> float:
        """Calculate composite accuracy score."""
        weights = {
            "precision": 0.15,
            "recall": 0.15,
            "f1_score": 0.20,
            "mrr": 0.25,
            "ndcg": 0.25,
        }
        return (
            weights["precision"] * self.precision +
            weights["recall"] * self.recall +
            weights["f1_score"] * self.f1_score +
            weights["mrr"] * self.mrr +
            weights["ndcg"] * self.ndcg
        )

    def get_efficiency_score(self) -> float:
        """
        Calculate efficiency score based on latency and resource usage.
        Lower latency and resource usage = higher score.
        """
        latency_score = max(0, 1 - (self.latency_ms / 1000))
        memory_score = max(0, 1 - (self.memory_mb / 10000))
        
        return 0.6 * latency_score + 0.4 * memory_score


class UnifiedScoreboard:
    """
    Unified scoreboard that aggregates metrics from all retrieval pipelines.
    
    This class provides methods to:
    - Add metrics from different pipelines
    - Load metrics from JSON files
    - Compare pipelines across different dimensions
    - Export aggregated results
    """

    def __init__(self):
        """Initialize the unified scoreboard."""
        self.pipelines: Dict[PipelineType, PipelineMetrics] = {}
        self.comparison_history: List[Dict[str, Any]] = []

    def add_pipeline_metrics(self, metrics: PipelineMetrics) -> None:
        """
        Add metrics for a pipeline.
        
        Args:
            metrics: PipelineMetrics object containing the metrics
        """
        self.pipelines[metrics.pipeline_type] = metrics

    def load_from_json(self, filepath: Union[str, Path]) -> None:
        """
        Load pipeline metrics from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing metrics
        """
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                metrics = PipelineMetrics.from_dict(item)
                self.add_pipeline_metrics(metrics)
        elif isinstance(data, dict):
            if "pipelines" in data:
                for item in data["pipelines"]:
                    metrics = PipelineMetrics.from_dict(item)
                    self.add_pipeline_metrics(metrics)
            else:
                metrics = PipelineMetrics.from_dict(data)
                self.add_pipeline_metrics(metrics)

    def load_from_dict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Load pipeline metrics from a dictionary or list of dictionaries.
        
        Args:
            data: Dictionary or list of dictionaries containing metrics
        """
        if isinstance(data, list):
            for item in data:
                metrics = PipelineMetrics.from_dict(item)
                self.add_pipeline_metrics(metrics)
        elif isinstance(data, dict):
            if "pipelines" in data:
                for item in data["pipelines"]:
                    metrics = PipelineMetrics.from_dict(item)
                    self.add_pipeline_metrics(metrics)
            else:
                metrics = PipelineMetrics.from_dict(data)
                self.add_pipeline_metrics(metrics)

    def get_pipeline_metrics(self, pipeline_type: PipelineType) -> Optional[PipelineMetrics]:
        """
        Get metrics for a specific pipeline.
        
        Args:
            pipeline_type: Type of the pipeline
            
        Returns:
            PipelineMetrics if found, None otherwise
        """
        return self.pipelines.get(pipeline_type)

    def get_all_metrics(self) -> Dict[PipelineType, PipelineMetrics]:
        """Get all pipeline metrics."""
        return self.pipelines.copy()

    def get_metric_comparison(self, metric_name: str) -> Dict[str, float]:
        """
        Compare a specific metric across all pipelines.
        
        Args:
            metric_name: Name of the metric to compare
            
        Returns:
            Dictionary mapping pipeline names to metric values
        """
        comparison = {}
        for pipeline_type, metrics in self.pipelines.items():
            if hasattr(metrics, metric_name):
                comparison[pipeline_type.value] = getattr(metrics, metric_name)
        return comparison

    def get_ranking(self, metric_name: str, ascending: bool = False) -> List[tuple]:
        """
        Get pipeline ranking based on a specific metric.
        
        Args:
            metric_name: Name of the metric to rank by
            ascending: If True, lower values are better (e.g., latency)
            
        Returns:
            List of tuples (pipeline_name, metric_value) sorted by rank
        """
        comparison = self.get_metric_comparison(metric_name)
        sorted_items = sorted(
            comparison.items(),
            key=lambda x: x[1],
            reverse=not ascending
        )
        return sorted_items

    def get_best_pipeline(self, metric_name: str, ascending: bool = False) -> Optional[str]:
        """
        Get the best pipeline for a specific metric.
        
        Args:
            metric_name: Name of the metric
            ascending: If True, lower values are better
            
        Returns:
            Name of the best pipeline, or None if no pipelines
        """
        ranking = self.get_ranking(metric_name, ascending)
        if ranking:
            return ranking[0][0]
        return None

    def get_accuracy_ranking(self) -> List[tuple]:
        """
        Get pipeline ranking based on composite accuracy score.
        
        Returns:
            List of tuples (pipeline_name, accuracy_score) sorted by rank
        """
        scores = {}
        for pipeline_type, metrics in self.pipelines.items():
            scores[pipeline_type.value] = metrics.get_accuracy_score()
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def get_efficiency_ranking(self) -> List[tuple]:
        """
        Get pipeline ranking based on efficiency score.
        
        Returns:
            List of tuples (pipeline_name, efficiency_score) sorted by rank
        """
        scores = {}
        for pipeline_type, metrics in self.pipelines.items():
            scores[pipeline_type.value] = metrics.get_efficiency_score()
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def get_summary_table(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate a summary table of all metrics for all pipelines.
        
        Returns:
            Dictionary with pipeline names as keys and metric dictionaries as values
        """
        summary = {}
        for pipeline_type, metrics in self.pipelines.items():
            summary[pipeline_type.value] = {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "mrr": metrics.mrr,
                "ndcg": metrics.ndcg,
                "map_score": metrics.map_score,
                "latency_ms": metrics.latency_ms,
                "throughput_qps": metrics.throughput_qps,
                "memory_mb": metrics.memory_mb,
                "accuracy_score": metrics.get_accuracy_score(),
                "efficiency_score": metrics.get_efficiency_score(),
            }
        return summary

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Export all metrics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        filepath = Path(filepath)
        data = {
            "pipelines": [
                metrics.to_dict() for metrics in self.pipelines.values()
            ],
            "summary": self.get_summary_table(),
            "rankings": {
                "accuracy": self.get_accuracy_ranking(),
                "efficiency": self.get_efficiency_ranking(),
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all metrics to a dictionary.
        
        Returns:
            Dictionary containing all metrics and rankings
        """
        return {
            "pipelines": [
                metrics.to_dict() for metrics in self.pipelines.values()
            ],
            "summary": self.get_summary_table(),
            "rankings": {
                "accuracy": self.get_accuracy_ranking(),
                "efficiency": self.get_efficiency_ranking(),
            }
        }

    def __len__(self) -> int:
        """Return number of pipelines in the scoreboard."""
        return len(self.pipelines)

    def __repr__(self) -> str:
        """Return string representation of the scoreboard."""
        pipeline_names = [p.value for p in self.pipelines.keys()]
        return f"UnifiedScoreboard(pipelines={pipeline_names})"
