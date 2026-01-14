"""
Retrieval Pipeline Comparison Dashboard

A comprehensive comparison dashboard that aggregates results from all retrieval
pipelines (BM25, Dense, Hybrid, ColBERT) and provides recommendations.
"""

from .scoreboard import UnifiedScoreboard, PipelineMetrics
from .visualization import VisualizationDashboard
from .recommendation import RecommendationEngine
from .report import ReportGenerator

__all__ = [
    "UnifiedScoreboard",
    "PipelineMetrics",
    "VisualizationDashboard",
    "RecommendationEngine",
    "ReportGenerator",
]

__version__ = "0.1.0"
