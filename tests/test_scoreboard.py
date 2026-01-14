"""Unit tests for scoreboard output module."""

import json
import os
import tempfile
import pytest
from src.scoreboard.output import Scoreboard, generate_recommendations


class TestScoreboard:
    """Tests for Scoreboard class."""

    def test_init_default_name(self):
        """Test scoreboard initialization with default name."""
        scoreboard = Scoreboard()
        assert scoreboard.experiment_name.startswith("experiment_")

    def test_init_custom_name(self):
        """Test scoreboard initialization with custom name."""
        scoreboard = Scoreboard(experiment_name="test_experiment")
        assert scoreboard.experiment_name == "test_experiment"

    def test_add_per_query_results(self):
        """Test adding per-query results."""
        scoreboard = Scoreboard()
        results = [
            {"query_id": "q1", "precision@1": 1.0},
            {"query_id": "q2", "precision@1": 0.5}
        ]
        
        scoreboard.add_per_query_results(results)
        assert len(scoreboard.per_query_results) == 2

    def test_add_aggregate_results(self):
        """Test adding aggregate results."""
        scoreboard = Scoreboard()
        aggregate = {"mean_precision@1": 0.75, "mean_mrr": 0.8}
        
        scoreboard.add_aggregate_results(aggregate)
        assert scoreboard.aggregate_results["mean_precision@1"] == 0.75

    def test_add_model_info(self):
        """Test adding model information."""
        scoreboard = Scoreboard()
        info = {"model_name": "test-model", "dimension": 384}
        
        scoreboard.add_model_info(info)
        assert scoreboard.model_info["model_name"] == "test-model"

    def test_add_dataset_info(self):
        """Test adding dataset information."""
        scoreboard = Scoreboard()
        info = {"num_documents": 100, "num_queries": 50}
        
        scoreboard.add_dataset_info(info)
        assert scoreboard.dataset_info["num_documents"] == 100

    def test_add_recommendations(self):
        """Test adding recommendations."""
        scoreboard = Scoreboard()
        recommendations = ["Use larger model", "Increase k"]
        
        scoreboard.add_recommendations(recommendations)
        assert len(scoreboard.recommendations) == 2

    def test_generate_summary(self):
        """Test generating summary."""
        scoreboard = Scoreboard(experiment_name="test")
        scoreboard.add_aggregate_results({"mean_mrr": 0.8})
        scoreboard.add_model_info({"model_name": "test-model"})
        
        summary = scoreboard.generate_summary()
        
        assert summary["experiment_name"] == "test"
        assert "timestamp" in summary
        assert summary["model_info"]["model_name"] == "test-model"

    def test_to_json_string(self):
        """Test JSON string generation."""
        scoreboard = Scoreboard(experiment_name="test")
        scoreboard.add_aggregate_results({"mean_mrr": 0.8})
        
        json_str = scoreboard.to_json()
        data = json.loads(json_str)
        
        assert data["experiment_name"] == "test"
        assert data["aggregate_metrics"]["mean_mrr"] == 0.8

    def test_to_json_file(self):
        """Test JSON file output."""
        scoreboard = Scoreboard(experiment_name="test")
        scoreboard.add_aggregate_results({"mean_mrr": 0.8})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            scoreboard.to_json(output_path)
            
            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert data["experiment_name"] == "test"

    def test_to_csv(self):
        """Test CSV file output."""
        scoreboard = Scoreboard()
        scoreboard.add_per_query_results([
            {"query_id": "q1", "precision@1": 1.0, "mrr": 1.0},
            {"query_id": "q2", "precision@1": 0.5, "mrr": 0.5}
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")
            scoreboard.to_csv(output_path)
            
            assert os.path.exists(output_path)
            with open(output_path) as f:
                content = f.read()
            assert "query_id" in content
            assert "q1" in content

    def test_to_aggregate_csv(self):
        """Test aggregate CSV file output."""
        scoreboard = Scoreboard()
        scoreboard.add_aggregate_results({
            "mean_precision@1": 0.75,
            "mean_mrr": 0.8
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "aggregate.csv")
            scoreboard.to_aggregate_csv(output_path)
            
            assert os.path.exists(output_path)
            with open(output_path) as f:
                content = f.read()
            assert "metric" in content
            assert "mean_precision@1" in content

    def test_print_summary(self):
        """Test print summary output."""
        scoreboard = Scoreboard(experiment_name="test")
        scoreboard.add_model_info({"model_name": "test-model", "dimension": 384})
        scoreboard.add_aggregate_results({"mean_mrr": 0.8})
        scoreboard.add_recommendations(["Test recommendation"])
        
        summary = scoreboard.print_summary()
        
        assert "RETRIEVAL EVALUATION SCOREBOARD" in summary
        assert "test-model" in summary
        assert "Test recommendation" in summary


class TestGenerateRecommendations:
    """Tests for generate_recommendations function."""

    def test_high_mrr_recommendation(self):
        """Test recommendation for high MRR."""
        metrics = {"mean_mrr": 0.85, "mean_precision@1": 0.8}
        recommendations = generate_recommendations(metrics, "all-MiniLM-L6-v2", 100)
        
        assert any("Excellent MRR" in r for r in recommendations)

    def test_low_mrr_recommendation(self):
        """Test recommendation for low MRR."""
        metrics = {"mean_mrr": 0.3, "mean_precision@1": 0.2}
        recommendations = generate_recommendations(metrics, "all-MiniLM-L6-v2", 100)
        
        assert any("Low MRR" in r for r in recommendations)

    def test_small_dataset_recommendation(self):
        """Test recommendation for small dataset."""
        metrics = {"mean_mrr": 0.7}
        recommendations = generate_recommendations(metrics, "all-MiniLM-L6-v2", 500)
        
        assert any("small datasets" in r.lower() or "flat index" in r.lower() for r in recommendations)

    def test_large_dataset_recommendation(self):
        """Test recommendation for large dataset."""
        metrics = {"mean_mrr": 0.7}
        recommendations = generate_recommendations(metrics, "all-MiniLM-L6-v2", 2000000)
        
        assert any("HNSW" in r for r in recommendations)

    def test_minilm_model_recommendation(self):
        """Test recommendation for MiniLM model."""
        metrics = {"mean_mrr": 0.7}
        recommendations = generate_recommendations(metrics, "all-MiniLM-L6-v2", 1000)
        
        assert any("MiniLM" in r for r in recommendations)

    def test_use_cases_recommendation(self):
        """Test that use cases recommendation is included."""
        metrics = {"mean_mrr": 0.7}
        recommendations = generate_recommendations(metrics, "test-model", 1000)
        
        assert any("Use Cases" in r or "semantic similarity" in r.lower() for r in recommendations)

    def test_high_precision_recommendation(self):
        """Test recommendation for high precision."""
        metrics = {"mean_mrr": 0.7, "mean_precision@1": 0.85}
        recommendations = generate_recommendations(metrics, "test-model", 1000)
        
        assert any("Precision@1" in r for r in recommendations)

    def test_low_recall_recommendation(self):
        """Test recommendation for low recall."""
        metrics = {"mean_mrr": 0.7, "mean_recall@10": 0.3}
        recommendations = generate_recommendations(metrics, "test-model", 1000)
        
        assert any("Recall@10" in r or "recall" in r.lower() for r in recommendations)
