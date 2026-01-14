"""
Example usage of the Retrieval Pipeline Comparison Dashboard.

This script demonstrates how to:
1. Create pipeline metrics
2. Aggregate them in a unified scoreboard
3. Generate visualizations
4. Get recommendations
5. Generate reports
"""

from pathlib import Path

from .scoreboard import UnifiedScoreboard, PipelineMetrics, PipelineType
from .visualization import VisualizationDashboard
from .recommendation import (
    RecommendationEngine,
    RequirementProfile,
    QueryType,
    LatencyRequirement,
    AccuracyRequirement,
    ResourceConstraint,
)
from .report import ReportGenerator


def create_sample_metrics() -> list:
    """Create sample metrics for demonstration."""
    
    bm25_metrics = PipelineMetrics(
        pipeline_type=PipelineType.BM25,
        precision=0.72,
        recall=0.68,
        f1_score=0.70,
        mrr=0.65,
        ndcg=0.71,
        ndcg_at_k={5: 0.68, 10: 0.71, 20: 0.73},
        map_score=0.67,
        latency_ms=15.2,
        latency_p50_ms=12.0,
        latency_p95_ms=25.0,
        latency_p99_ms=45.0,
        throughput_qps=850,
        memory_mb=512,
        index_size_mb=1024,
        cpu_utilization=25.0,
        gpu_utilization=None,
        num_queries=10000,
        metadata={"version": "1.0", "tokenizer": "standard"},
    )
    
    dense_metrics = PipelineMetrics(
        pipeline_type=PipelineType.DENSE,
        precision=0.85,
        recall=0.82,
        f1_score=0.83,
        mrr=0.80,
        ndcg=0.84,
        ndcg_at_k={5: 0.81, 10: 0.84, 20: 0.86},
        map_score=0.79,
        latency_ms=85.5,
        latency_p50_ms=75.0,
        latency_p95_ms=120.0,
        latency_p99_ms=180.0,
        throughput_qps=120,
        memory_mb=4096,
        index_size_mb=8192,
        cpu_utilization=45.0,
        gpu_utilization=65.0,
        num_queries=10000,
        metadata={"model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384},
    )
    
    hybrid_metrics = PipelineMetrics(
        pipeline_type=PipelineType.HYBRID,
        precision=0.88,
        recall=0.85,
        f1_score=0.86,
        mrr=0.83,
        ndcg=0.87,
        ndcg_at_k={5: 0.84, 10: 0.87, 20: 0.89},
        map_score=0.82,
        latency_ms=55.0,
        latency_p50_ms=48.0,
        latency_p95_ms=85.0,
        latency_p99_ms=120.0,
        throughput_qps=200,
        memory_mb=2048,
        index_size_mb=4096,
        cpu_utilization=40.0,
        gpu_utilization=35.0,
        num_queries=10000,
        metadata={"bm25_weight": 0.3, "dense_weight": 0.7},
    )
    
    colbert_metrics = PipelineMetrics(
        pipeline_type=PipelineType.COLBERT,
        precision=0.92,
        recall=0.89,
        f1_score=0.90,
        mrr=0.88,
        ndcg=0.91,
        ndcg_at_k={5: 0.88, 10: 0.91, 20: 0.93},
        map_score=0.87,
        latency_ms=150.0,
        latency_p50_ms=130.0,
        latency_p95_ms=220.0,
        latency_p99_ms=350.0,
        throughput_qps=65,
        memory_mb=8192,
        index_size_mb=16384,
        cpu_utilization=55.0,
        gpu_utilization=85.0,
        num_queries=10000,
        metadata={"model": "colbert-ir/colbertv2.0", "dim": 128},
    )
    
    return [bm25_metrics, dense_metrics, hybrid_metrics, colbert_metrics]


def run_example(output_dir: str = "./output") -> dict:
    """
    Run the complete example workflow.
    
    Args:
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating unified scoreboard...")
    scoreboard = UnifiedScoreboard()
    
    for metrics in create_sample_metrics():
        scoreboard.add_pipeline_metrics(metrics)
    
    print(f"Loaded {len(scoreboard)} pipelines")
    
    print("\nSummary Table:")
    summary = scoreboard.get_summary_table()
    for pipeline, metrics in summary.items():
        print(f"  {pipeline.upper()}: accuracy={metrics['accuracy_score']:.3f}, "
              f"efficiency={metrics['efficiency_score']:.3f}")
    
    print("\nCreating visualizations...")
    viz = VisualizationDashboard(scoreboard)
    
    viz_paths = viz.create_comprehensive_dashboard(
        output_path / "visualizations",
        use_plotly=True,
    )
    print(f"  Created {len(viz_paths)} visualizations")
    
    print("\nGenerating recommendations...")
    
    profile = RequirementProfile(
        query_type=QueryType.MIXED,
        latency_requirement=LatencyRequirement.LOW_LATENCY,
        accuracy_requirement=AccuracyRequirement.HIGH,
        resource_constraint=ResourceConstraint.MODERATE,
        max_latency_ms=100,
        min_accuracy=0.75,
    )
    
    engine = RecommendationEngine(scoreboard)
    recommendations = engine.get_recommendation(profile)
    
    print("\nPipeline Rankings:")
    for rec in recommendations:
        status = "MEETS REQUIREMENTS" if rec.meets_requirements else "DOES NOT MEET"
        print(f"  {rec.rank}. {rec.pipeline_type.upper()} "
              f"(score: {rec.overall_score:.3f}) - {status}")
    
    best = engine.get_best_pipeline(profile)
    if best:
        print(f"\nBest Pipeline: {best.pipeline_type.upper()}")
        print(f"  Reasoning: {best.reasoning}")
    
    print("\nCost-Benefit Analysis:")
    for pipeline, analysis in engine.get_all_cost_benefit_analyses().items():
        print(f"  {pipeline.upper()}: net_benefit={analysis.net_benefit:.3f}, "
              f"roi={analysis.roi_score:.2f}")
    
    print("\nGenerating report...")
    report = ReportGenerator(scoreboard, profile)
    
    report_paths = report.save_all(output_path / "reports")
    print(f"  Saved reports to {output_path / 'reports'}")
    
    scoreboard.export_to_json(output_path / "scoreboard.json")
    engine.export_recommendations(profile, output_path / "recommendations.json")
    
    print("\nExample completed successfully!")
    print(f"Output files saved to: {output_path}")
    
    return {
        "scoreboard": str(output_path / "scoreboard.json"),
        "recommendations": str(output_path / "recommendations.json"),
        "visualizations": viz_paths,
        "reports": report_paths,
    }


def load_from_json_example(json_path: str) -> UnifiedScoreboard:
    """
    Example of loading metrics from a JSON file.
    
    Args:
        json_path: Path to JSON file with pipeline metrics
        
    Returns:
        UnifiedScoreboard with loaded metrics
    """
    scoreboard = UnifiedScoreboard()
    scoreboard.load_from_json(json_path)
    return scoreboard


def custom_profile_example() -> RequirementProfile:
    """
    Example of creating a custom requirement profile.
    
    Returns:
        RequirementProfile for real-time semantic search
    """
    return RequirementProfile(
        query_type=QueryType.SEMANTIC,
        latency_requirement=LatencyRequirement.REAL_TIME,
        accuracy_requirement=AccuracyRequirement.CRITICAL,
        resource_constraint=ResourceConstraint.UNLIMITED,
        max_latency_ms=20,
        min_accuracy=0.85,
        requires_gpu=True,
        custom_weights={
            "query_type": 0.30,
            "latency": 0.35,
            "accuracy": 0.25,
            "resource": 0.10,
        },
    )


if __name__ == "__main__":
    run_example()
