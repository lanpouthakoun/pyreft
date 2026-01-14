"""
Tests for the retrieval_comparison module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from retrieval_comparison.scoreboard import (
    UnifiedScoreboard,
    PipelineMetrics,
    PipelineType,
)
from retrieval_comparison.recommendation import (
    RecommendationEngine,
    RequirementProfile,
    QueryType,
    LatencyRequirement,
    AccuracyRequirement,
    ResourceConstraint,
)
from retrieval_comparison.report import ReportGenerator


def create_sample_metrics():
    """Create sample metrics for testing."""
    bm25 = PipelineMetrics(
        pipeline_type=PipelineType.BM25,
        precision=0.72,
        recall=0.68,
        f1_score=0.70,
        mrr=0.65,
        ndcg=0.71,
        latency_ms=15.0,
        throughput_qps=850,
        memory_mb=512,
    )
    
    dense = PipelineMetrics(
        pipeline_type=PipelineType.DENSE,
        precision=0.85,
        recall=0.82,
        f1_score=0.83,
        mrr=0.80,
        ndcg=0.84,
        latency_ms=85.0,
        throughput_qps=120,
        memory_mb=4096,
    )
    
    hybrid = PipelineMetrics(
        pipeline_type=PipelineType.HYBRID,
        precision=0.88,
        recall=0.85,
        f1_score=0.86,
        mrr=0.83,
        ndcg=0.87,
        latency_ms=55.0,
        throughput_qps=200,
        memory_mb=2048,
    )
    
    colbert = PipelineMetrics(
        pipeline_type=PipelineType.COLBERT,
        precision=0.92,
        recall=0.89,
        f1_score=0.90,
        mrr=0.88,
        ndcg=0.91,
        latency_ms=150.0,
        throughput_qps=65,
        memory_mb=8192,
    )
    
    return [bm25, dense, hybrid, colbert]


class TestPipelineMetrics:
    """Tests for PipelineMetrics class."""
    
    def test_create_metrics(self):
        """Test creating pipeline metrics."""
        metrics = PipelineMetrics(
            pipeline_type=PipelineType.BM25,
            precision=0.75,
            recall=0.70,
        )
        
        assert metrics.pipeline_type == PipelineType.BM25
        assert metrics.precision == 0.75
        assert metrics.recall == 0.70
        assert metrics.f1_score == 0.0
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PipelineMetrics(
            pipeline_type=PipelineType.DENSE,
            precision=0.85,
            recall=0.82,
            latency_ms=50.0,
        )
        
        data = metrics.to_dict()
        
        assert data["pipeline_type"] == "dense"
        assert data["precision"] == 0.85
        assert data["recall"] == 0.82
        assert data["latency_ms"] == 50.0
    
    def test_from_dict(self):
        """Test creating metrics from dictionary."""
        data = {
            "pipeline_type": "hybrid",
            "precision": 0.88,
            "recall": 0.85,
            "f1_score": 0.86,
            "latency_ms": 55.0,
        }
        
        metrics = PipelineMetrics.from_dict(data)
        
        assert metrics.pipeline_type == PipelineType.HYBRID
        assert metrics.precision == 0.88
        assert metrics.recall == 0.85
        assert metrics.f1_score == 0.86
        assert metrics.latency_ms == 55.0
    
    def test_accuracy_score(self):
        """Test composite accuracy score calculation."""
        metrics = PipelineMetrics(
            pipeline_type=PipelineType.BM25,
            precision=0.80,
            recall=0.80,
            f1_score=0.80,
            mrr=0.80,
            ndcg=0.80,
        )
        
        score = metrics.get_accuracy_score()
        
        assert 0.79 <= score <= 0.81
    
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        metrics = PipelineMetrics(
            pipeline_type=PipelineType.BM25,
            latency_ms=100.0,
            memory_mb=1000.0,
        )
        
        score = metrics.get_efficiency_score()
        
        assert 0 <= score <= 1


class TestUnifiedScoreboard:
    """Tests for UnifiedScoreboard class."""
    
    def test_add_pipeline_metrics(self):
        """Test adding pipeline metrics."""
        scoreboard = UnifiedScoreboard()
        metrics = PipelineMetrics(
            pipeline_type=PipelineType.BM25,
            precision=0.75,
        )
        
        scoreboard.add_pipeline_metrics(metrics)
        
        assert len(scoreboard) == 1
        assert PipelineType.BM25 in scoreboard.pipelines
    
    def test_get_pipeline_metrics(self):
        """Test getting pipeline metrics."""
        scoreboard = UnifiedScoreboard()
        metrics = PipelineMetrics(
            pipeline_type=PipelineType.DENSE,
            precision=0.85,
        )
        scoreboard.add_pipeline_metrics(metrics)
        
        retrieved = scoreboard.get_pipeline_metrics(PipelineType.DENSE)
        
        assert retrieved is not None
        assert retrieved.precision == 0.85
    
    def test_get_metric_comparison(self):
        """Test comparing metrics across pipelines."""
        scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            scoreboard.add_pipeline_metrics(metrics)
        
        comparison = scoreboard.get_metric_comparison("precision")
        
        assert len(comparison) == 4
        assert "bm25" in comparison
        assert "dense" in comparison
        assert comparison["bm25"] == 0.72
        assert comparison["dense"] == 0.85
    
    def test_get_ranking(self):
        """Test getting pipeline ranking."""
        scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            scoreboard.add_pipeline_metrics(metrics)
        
        ranking = scoreboard.get_ranking("precision")
        
        assert len(ranking) == 4
        assert ranking[0][0] == "colbert"
        assert ranking[-1][0] == "bm25"
    
    def test_get_best_pipeline(self):
        """Test getting best pipeline for a metric."""
        scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            scoreboard.add_pipeline_metrics(metrics)
        
        best = scoreboard.get_best_pipeline("precision")
        
        assert best == "colbert"
    
    def test_get_best_pipeline_ascending(self):
        """Test getting best pipeline with ascending order."""
        scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            scoreboard.add_pipeline_metrics(metrics)
        
        best = scoreboard.get_best_pipeline("latency_ms", ascending=True)
        
        assert best == "bm25"
    
    def test_get_summary_table(self):
        """Test generating summary table."""
        scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            scoreboard.add_pipeline_metrics(metrics)
        
        summary = scoreboard.get_summary_table()
        
        assert len(summary) == 4
        assert "bm25" in summary
        assert "precision" in summary["bm25"]
        assert "accuracy_score" in summary["bm25"]
        assert "efficiency_score" in summary["bm25"]
    
    def test_load_from_dict(self):
        """Test loading metrics from dictionary."""
        scoreboard = UnifiedScoreboard()
        data = [
            {"pipeline_type": "bm25", "precision": 0.72},
            {"pipeline_type": "dense", "precision": 0.85},
        ]
        
        scoreboard.load_from_dict(data)
        
        assert len(scoreboard) == 2
    
    def test_export_to_json(self):
        """Test exporting to JSON file."""
        scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            scoreboard.add_pipeline_metrics(metrics)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            scoreboard.export_to_json(filepath)
            
            with open(filepath) as f:
                data = json.load(f)
            
            assert "pipelines" in data
            assert "summary" in data
            assert "rankings" in data
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestRecommendationEngine:
    """Tests for RecommendationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            self.scoreboard.add_pipeline_metrics(metrics)
        self.engine = RecommendationEngine(self.scoreboard)
    
    def test_get_recommendation(self):
        """Test getting recommendations."""
        profile = RequirementProfile()
        
        recommendations = self.engine.get_recommendation(profile)
        
        assert len(recommendations) == 4
        assert recommendations[0].rank == 1
        assert recommendations[-1].rank == 4
    
    def test_get_recommendation_with_latency_requirement(self):
        """Test recommendations with latency requirement."""
        profile = RequirementProfile(
            latency_requirement=LatencyRequirement.LOW_LATENCY,
            max_latency_ms=50,
        )
        
        recommendations = self.engine.get_recommendation(profile)
        
        bm25_rec = next(r for r in recommendations if r.pipeline_type == "bm25")
        assert bm25_rec.meets_requirements is True
        
        colbert_rec = next(r for r in recommendations if r.pipeline_type == "colbert")
        assert colbert_rec.meets_requirements is False
    
    def test_get_recommendation_with_accuracy_requirement(self):
        """Test recommendations with accuracy requirement."""
        profile = RequirementProfile(
            accuracy_requirement=AccuracyRequirement.CRITICAL,
            min_accuracy=0.85,
        )
        
        recommendations = self.engine.get_recommendation(profile)
        
        colbert_rec = next(r for r in recommendations if r.pipeline_type == "colbert")
        assert colbert_rec.meets_requirements is True
    
    def test_get_recommendation_query_type_keyword(self):
        """Test recommendations for keyword-heavy queries."""
        profile = RequirementProfile(
            query_type=QueryType.KEYWORD_HEAVY,
        )
        
        recommendations = self.engine.get_recommendation(profile)
        
        bm25_score = next(r for r in recommendations if r.pipeline_type == "bm25")
        dense_score = next(r for r in recommendations if r.pipeline_type == "dense")
        
        assert bm25_score.detailed_scores["query_type"] > dense_score.detailed_scores["query_type"]
    
    def test_get_recommendation_query_type_semantic(self):
        """Test recommendations for semantic queries."""
        profile = RequirementProfile(
            query_type=QueryType.SEMANTIC,
        )
        
        recommendations = self.engine.get_recommendation(profile)
        
        bm25_score = next(r for r in recommendations if r.pipeline_type == "bm25")
        colbert_score = next(r for r in recommendations if r.pipeline_type == "colbert")
        
        assert colbert_score.detailed_scores["query_type"] > bm25_score.detailed_scores["query_type"]
    
    def test_get_best_pipeline(self):
        """Test getting best pipeline."""
        profile = RequirementProfile()
        
        best = self.engine.get_best_pipeline(profile)
        
        assert best is not None
        assert best.rank == 1
    
    def test_calculate_cost_benefit(self):
        """Test cost-benefit analysis."""
        metrics = self.scoreboard.get_pipeline_metrics(PipelineType.BM25)
        
        analysis = self.engine.calculate_cost_benefit(metrics)
        
        assert analysis.pipeline_type == "bm25"
        assert 0 <= analysis.accuracy_benefit <= 1
        assert 0 <= analysis.latency_cost <= 1
        assert 0 <= analysis.resource_cost <= 1
        assert len(analysis.recommendations) > 0
    
    def test_get_all_cost_benefit_analyses(self):
        """Test getting all cost-benefit analyses."""
        analyses = self.engine.get_all_cost_benefit_analyses()
        
        assert len(analyses) == 4
        assert "bm25" in analyses
        assert "dense" in analyses


class TestRequirementProfile:
    """Tests for RequirementProfile class."""
    
    def test_default_profile(self):
        """Test default profile values."""
        profile = RequirementProfile()
        
        assert profile.query_type == QueryType.MIXED
        assert profile.latency_requirement == LatencyRequirement.MODERATE
        assert profile.accuracy_requirement == AccuracyRequirement.HIGH
        assert profile.resource_constraint == ResourceConstraint.MODERATE
    
    def test_to_dict(self):
        """Test converting profile to dictionary."""
        profile = RequirementProfile(
            query_type=QueryType.SEMANTIC,
            max_latency_ms=100,
        )
        
        data = profile.to_dict()
        
        assert data["query_type"] == "semantic"
        assert data["max_latency_ms"] == 100
    
    def test_from_dict(self):
        """Test creating profile from dictionary."""
        data = {
            "query_type": "keyword_heavy",
            "latency_requirement": "real_time",
            "max_latency_ms": 20,
        }
        
        profile = RequirementProfile.from_dict(data)
        
        assert profile.query_type == QueryType.KEYWORD_HEAVY
        assert profile.latency_requirement == LatencyRequirement.REAL_TIME
        assert profile.max_latency_ms == 20


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scoreboard = UnifiedScoreboard()
        for metrics in create_sample_metrics():
            self.scoreboard.add_pipeline_metrics(metrics)
        self.profile = RequirementProfile()
        self.report = ReportGenerator(self.scoreboard, self.profile)
    
    def test_generate_report(self):
        """Test generating report sections."""
        sections = self.report.generate_report()
        
        assert len(sections) > 0
        assert any("Executive Summary" in s.title for s in sections)
        assert any("Recommendations" in s.title for s in sections)
    
    def test_to_markdown(self):
        """Test converting report to markdown."""
        markdown = self.report.to_markdown()
        
        assert "# Retrieval Pipeline Comparison Report" in markdown
        assert "Executive Summary" in markdown
        assert "Recommendations" in markdown
    
    def test_to_dict(self):
        """Test converting report to dictionary."""
        data = self.report.to_dict()
        
        assert "generated_at" in data
        assert "profile" in data
        assert "scoreboard" in data
        assert "recommendations" in data
        assert "sections" in data
    
    def test_save_markdown(self):
        """Test saving report as markdown."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            filepath = f.name
        
        try:
            self.report.save_markdown(filepath)
            
            with open(filepath) as f:
                content = f.read()
            
            assert "Retrieval Pipeline Comparison Report" in content
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_save_json(self):
        """Test saving report as JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            self.report.save_json(filepath)
            
            with open(filepath) as f:
                data = json.load(f)
            
            assert "generated_at" in data
            assert "sections" in data
        finally:
            Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
