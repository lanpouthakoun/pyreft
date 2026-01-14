"""
Recommendation Engine Module

Builds logic that recommends the best pipeline based on:
- Query type (keyword-heavy vs semantic)
- Latency requirements
- Accuracy requirements
- Resource constraints

Includes cost-benefit analysis for each pipeline type.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from pathlib import Path

from .scoreboard import UnifiedScoreboard, PipelineMetrics, PipelineType


class QueryType(Enum):
    """Types of queries for recommendation."""
    KEYWORD_HEAVY = "keyword_heavy"
    SEMANTIC = "semantic"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LatencyRequirement(Enum):
    """Latency requirement levels."""
    REAL_TIME = "real_time"
    LOW_LATENCY = "low_latency"
    MODERATE = "moderate"
    BATCH = "batch"


class AccuracyRequirement(Enum):
    """Accuracy requirement levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class ResourceConstraint(Enum):
    """Resource constraint levels."""
    MINIMAL = "minimal"
    LIMITED = "limited"
    MODERATE = "moderate"
    UNLIMITED = "unlimited"


@dataclass
class RequirementProfile:
    """
    Profile of requirements for pipeline selection.
    
    Attributes:
        query_type: Type of queries (keyword-heavy, semantic, mixed)
        latency_requirement: Latency requirement level
        accuracy_requirement: Accuracy requirement level
        resource_constraint: Resource constraint level
        max_latency_ms: Maximum acceptable latency in ms
        min_accuracy: Minimum acceptable accuracy score (0-1)
        max_memory_mb: Maximum acceptable memory usage in MB
        requires_gpu: Whether GPU is available/required
        custom_weights: Custom weights for scoring
    """
    query_type: QueryType = QueryType.MIXED
    latency_requirement: LatencyRequirement = LatencyRequirement.MODERATE
    accuracy_requirement: AccuracyRequirement = AccuracyRequirement.HIGH
    resource_constraint: ResourceConstraint = ResourceConstraint.MODERATE
    max_latency_ms: Optional[float] = None
    min_accuracy: Optional[float] = None
    max_memory_mb: Optional[float] = None
    requires_gpu: bool = False
    custom_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "query_type": self.query_type.value,
            "latency_requirement": self.latency_requirement.value,
            "accuracy_requirement": self.accuracy_requirement.value,
            "resource_constraint": self.resource_constraint.value,
            "max_latency_ms": self.max_latency_ms,
            "min_accuracy": self.min_accuracy,
            "max_memory_mb": self.max_memory_mb,
            "requires_gpu": self.requires_gpu,
            "custom_weights": self.custom_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementProfile":
        """Create profile from dictionary."""
        return cls(
            query_type=QueryType(data.get("query_type", "mixed")),
            latency_requirement=LatencyRequirement(data.get("latency_requirement", "moderate")),
            accuracy_requirement=AccuracyRequirement(data.get("accuracy_requirement", "high")),
            resource_constraint=ResourceConstraint(data.get("resource_constraint", "moderate")),
            max_latency_ms=data.get("max_latency_ms"),
            min_accuracy=data.get("min_accuracy"),
            max_memory_mb=data.get("max_memory_mb"),
            requires_gpu=data.get("requires_gpu", False),
            custom_weights=data.get("custom_weights", {}),
        )


@dataclass
class PipelineRecommendation:
    """
    Recommendation for a specific pipeline.
    
    Attributes:
        pipeline_type: Type of the recommended pipeline
        overall_score: Overall recommendation score (0-1)
        rank: Rank among all pipelines (1 = best)
        meets_requirements: Whether the pipeline meets all requirements
        strengths: List of pipeline strengths
        weaknesses: List of pipeline weaknesses
        cost_benefit_score: Cost-benefit analysis score
        detailed_scores: Breakdown of scores by category
        reasoning: Explanation for the recommendation
    """
    pipeline_type: str
    overall_score: float
    rank: int
    meets_requirements: bool
    strengths: List[str]
    weaknesses: List[str]
    cost_benefit_score: float
    detailed_scores: Dict[str, float]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            "pipeline_type": self.pipeline_type,
            "overall_score": self.overall_score,
            "rank": self.rank,
            "meets_requirements": self.meets_requirements,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "cost_benefit_score": self.cost_benefit_score,
            "detailed_scores": self.detailed_scores,
            "reasoning": self.reasoning,
        }


@dataclass
class CostBenefitAnalysis:
    """
    Cost-benefit analysis for a pipeline.
    
    Attributes:
        pipeline_type: Type of the pipeline
        accuracy_benefit: Benefit from accuracy (0-1)
        latency_cost: Cost from latency (0-1, lower is better)
        resource_cost: Cost from resource usage (0-1, lower is better)
        complexity_cost: Cost from implementation complexity (0-1)
        net_benefit: Net benefit score
        roi_score: Return on investment score
        recommendations: Specific recommendations for this pipeline
    """
    pipeline_type: str
    accuracy_benefit: float
    latency_cost: float
    resource_cost: float
    complexity_cost: float
    net_benefit: float
    roi_score: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "pipeline_type": self.pipeline_type,
            "accuracy_benefit": self.accuracy_benefit,
            "latency_cost": self.latency_cost,
            "resource_cost": self.resource_cost,
            "complexity_cost": self.complexity_cost,
            "net_benefit": self.net_benefit,
            "roi_score": self.roi_score,
            "recommendations": self.recommendations,
        }


class RecommendationEngine:
    """
    Recommendation engine for selecting the best retrieval pipeline.
    
    Analyzes pipeline metrics against requirements and provides
    recommendations with cost-benefit analysis.
    """

    PIPELINE_CHARACTERISTICS = {
        "bm25": {
            "query_affinity": {
                QueryType.KEYWORD_HEAVY: 1.0,
                QueryType.SEMANTIC: 0.3,
                QueryType.MIXED: 0.6,
                QueryType.UNKNOWN: 0.5,
            },
            "typical_latency": "low",
            "typical_accuracy": "moderate",
            "resource_usage": "low",
            "complexity": 0.2,
            "requires_gpu": False,
            "strengths": [
                "Fast query processing",
                "Low resource requirements",
                "Excellent for exact keyword matching",
                "Simple to deploy and maintain",
                "No training required",
            ],
            "weaknesses": [
                "Limited semantic understanding",
                "Struggles with synonyms and paraphrases",
                "No contextual understanding",
                "Vocabulary mismatch issues",
            ],
        },
        "dense": {
            "query_affinity": {
                QueryType.KEYWORD_HEAVY: 0.5,
                QueryType.SEMANTIC: 0.95,
                QueryType.MIXED: 0.75,
                QueryType.UNKNOWN: 0.7,
            },
            "typical_latency": "moderate",
            "typical_accuracy": "high",
            "resource_usage": "high",
            "complexity": 0.7,
            "requires_gpu": True,
            "strengths": [
                "Excellent semantic understanding",
                "Handles synonyms and paraphrases well",
                "Good for natural language queries",
                "Captures contextual meaning",
            ],
            "weaknesses": [
                "Higher latency than BM25",
                "Requires GPU for optimal performance",
                "Needs training data",
                "Higher memory requirements",
            ],
        },
        "hybrid": {
            "query_affinity": {
                QueryType.KEYWORD_HEAVY: 0.85,
                QueryType.SEMANTIC: 0.85,
                QueryType.MIXED: 0.95,
                QueryType.UNKNOWN: 0.85,
            },
            "typical_latency": "moderate",
            "typical_accuracy": "high",
            "resource_usage": "moderate",
            "complexity": 0.6,
            "requires_gpu": True,
            "strengths": [
                "Best of both worlds (lexical + semantic)",
                "Robust across query types",
                "Good balance of speed and accuracy",
                "Handles diverse query patterns",
            ],
            "weaknesses": [
                "More complex to tune",
                "Moderate resource requirements",
                "Requires balancing multiple components",
                "May not excel at extremes",
            ],
        },
        "colbert": {
            "query_affinity": {
                QueryType.KEYWORD_HEAVY: 0.7,
                QueryType.SEMANTIC: 0.98,
                QueryType.MIXED: 0.9,
                QueryType.UNKNOWN: 0.8,
            },
            "typical_latency": "high",
            "typical_accuracy": "very_high",
            "resource_usage": "very_high",
            "complexity": 0.9,
            "requires_gpu": True,
            "strengths": [
                "State-of-the-art accuracy",
                "Fine-grained token-level matching",
                "Excellent for complex queries",
                "Strong semantic understanding",
            ],
            "weaknesses": [
                "Highest latency",
                "Very high resource requirements",
                "Complex to deploy",
                "Large index sizes",
            ],
        },
    }

    LATENCY_THRESHOLDS = {
        LatencyRequirement.REAL_TIME: 10,
        LatencyRequirement.LOW_LATENCY: 50,
        LatencyRequirement.MODERATE: 200,
        LatencyRequirement.BATCH: 1000,
    }

    ACCURACY_THRESHOLDS = {
        AccuracyRequirement.CRITICAL: 0.9,
        AccuracyRequirement.HIGH: 0.75,
        AccuracyRequirement.MODERATE: 0.6,
        AccuracyRequirement.LOW: 0.4,
    }

    RESOURCE_THRESHOLDS = {
        ResourceConstraint.MINIMAL: 500,
        ResourceConstraint.LIMITED: 2000,
        ResourceConstraint.MODERATE: 8000,
        ResourceConstraint.UNLIMITED: float("inf"),
    }

    def __init__(self, scoreboard: UnifiedScoreboard):
        """
        Initialize the recommendation engine.
        
        Args:
            scoreboard: UnifiedScoreboard containing pipeline metrics
        """
        self.scoreboard = scoreboard

    def _get_latency_score(
        self,
        metrics: PipelineMetrics,
        profile: RequirementProfile,
    ) -> float:
        """Calculate latency score based on requirements."""
        if profile.max_latency_ms is not None:
            threshold = profile.max_latency_ms
        else:
            threshold = self.LATENCY_THRESHOLDS.get(
                profile.latency_requirement,
                200,
            )
        
        if metrics.latency_ms <= threshold:
            return 1.0 - (metrics.latency_ms / threshold) * 0.3
        else:
            return max(0, 0.7 - (metrics.latency_ms - threshold) / threshold)

    def _get_accuracy_score(
        self,
        metrics: PipelineMetrics,
        profile: RequirementProfile,
    ) -> float:
        """Calculate accuracy score based on requirements."""
        if profile.min_accuracy is not None:
            threshold = profile.min_accuracy
        else:
            threshold = self.ACCURACY_THRESHOLDS.get(
                profile.accuracy_requirement,
                0.75,
            )
        
        accuracy = metrics.get_accuracy_score()
        
        if accuracy >= threshold:
            return 0.7 + (accuracy - threshold) / (1 - threshold) * 0.3
        else:
            return accuracy / threshold * 0.7

    def _get_resource_score(
        self,
        metrics: PipelineMetrics,
        profile: RequirementProfile,
    ) -> float:
        """Calculate resource score based on constraints."""
        if profile.max_memory_mb is not None:
            threshold = profile.max_memory_mb
        else:
            threshold = self.RESOURCE_THRESHOLDS.get(
                profile.resource_constraint,
                8000,
            )
        
        if metrics.memory_mb <= threshold:
            return 1.0 - (metrics.memory_mb / threshold) * 0.3
        else:
            return max(0, 0.7 - (metrics.memory_mb - threshold) / threshold)

    def _get_query_type_score(
        self,
        pipeline_type: str,
        profile: RequirementProfile,
    ) -> float:
        """Calculate query type affinity score."""
        characteristics = self.PIPELINE_CHARACTERISTICS.get(pipeline_type, {})
        query_affinity = characteristics.get("query_affinity", {})
        return query_affinity.get(profile.query_type, 0.5)

    def _check_requirements_met(
        self,
        metrics: PipelineMetrics,
        profile: RequirementProfile,
    ) -> bool:
        """Check if pipeline meets all hard requirements."""
        if profile.max_latency_ms is not None:
            if metrics.latency_ms > profile.max_latency_ms:
                return False
        
        if profile.min_accuracy is not None:
            if metrics.get_accuracy_score() < profile.min_accuracy:
                return False
        
        if profile.max_memory_mb is not None:
            if metrics.memory_mb > profile.max_memory_mb:
                return False
        
        pipeline_type = metrics.pipeline_type.value
        characteristics = self.PIPELINE_CHARACTERISTICS.get(pipeline_type, {})
        
        if profile.requires_gpu and not characteristics.get("requires_gpu", False):
            pass
        
        if not profile.requires_gpu and characteristics.get("requires_gpu", False):
            if profile.resource_constraint == ResourceConstraint.MINIMAL:
                return False
        
        return True

    def _get_strengths_weaknesses(
        self,
        pipeline_type: str,
        metrics: PipelineMetrics,
        profile: RequirementProfile,
    ) -> tuple:
        """Get relevant strengths and weaknesses for the pipeline."""
        characteristics = self.PIPELINE_CHARACTERISTICS.get(pipeline_type, {})
        
        strengths = list(characteristics.get("strengths", []))
        weaknesses = list(characteristics.get("weaknesses", []))
        
        accuracy = metrics.get_accuracy_score()
        if accuracy >= 0.8:
            strengths.append(f"High accuracy score ({accuracy:.2f})")
        elif accuracy < 0.5:
            weaknesses.append(f"Low accuracy score ({accuracy:.2f})")
        
        if metrics.latency_ms < 50:
            strengths.append(f"Very low latency ({metrics.latency_ms:.1f}ms)")
        elif metrics.latency_ms > 200:
            weaknesses.append(f"High latency ({metrics.latency_ms:.1f}ms)")
        
        if metrics.throughput_qps > 100:
            strengths.append(f"High throughput ({metrics.throughput_qps:.0f} QPS)")
        
        return strengths[:5], weaknesses[:4]

    def _generate_reasoning(
        self,
        pipeline_type: str,
        metrics: PipelineMetrics,
        profile: RequirementProfile,
        scores: Dict[str, float],
    ) -> str:
        """Generate reasoning for the recommendation."""
        parts = []
        
        query_score = scores.get("query_type", 0)
        if query_score >= 0.8:
            parts.append(
                f"{pipeline_type.upper()} is well-suited for {profile.query_type.value} queries"
            )
        elif query_score < 0.5:
            parts.append(
                f"{pipeline_type.upper()} may not be optimal for {profile.query_type.value} queries"
            )
        
        latency_score = scores.get("latency", 0)
        if latency_score >= 0.8:
            parts.append(f"meets latency requirements ({metrics.latency_ms:.1f}ms)")
        elif latency_score < 0.5:
            parts.append(f"latency may be a concern ({metrics.latency_ms:.1f}ms)")
        
        accuracy_score = scores.get("accuracy", 0)
        if accuracy_score >= 0.8:
            parts.append(f"provides high accuracy ({metrics.get_accuracy_score():.2f})")
        elif accuracy_score < 0.5:
            parts.append(f"accuracy may not meet requirements")
        
        resource_score = scores.get("resource", 0)
        if resource_score >= 0.8:
            parts.append("efficient resource usage")
        elif resource_score < 0.5:
            parts.append("high resource requirements")
        
        if parts:
            return "; ".join(parts) + "."
        return f"{pipeline_type.upper()} is a viable option for the given requirements."

    def calculate_cost_benefit(
        self,
        metrics: PipelineMetrics,
    ) -> CostBenefitAnalysis:
        """
        Calculate cost-benefit analysis for a pipeline.
        
        Args:
            metrics: Pipeline metrics
            
        Returns:
            CostBenefitAnalysis object
        """
        pipeline_type = metrics.pipeline_type.value
        characteristics = self.PIPELINE_CHARACTERISTICS.get(pipeline_type, {})
        
        accuracy_benefit = metrics.get_accuracy_score()
        
        latency_cost = min(1.0, metrics.latency_ms / 500)
        
        resource_cost = min(1.0, metrics.memory_mb / 10000)
        
        complexity_cost = characteristics.get("complexity", 0.5)
        
        benefit_weight = 0.5
        cost_weight = 0.5
        
        total_cost = (
            0.4 * latency_cost +
            0.4 * resource_cost +
            0.2 * complexity_cost
        )
        
        net_benefit = benefit_weight * accuracy_benefit - cost_weight * total_cost
        
        if total_cost > 0:
            roi_score = accuracy_benefit / (total_cost + 0.1)
        else:
            roi_score = accuracy_benefit * 10
        
        recommendations = []
        
        if latency_cost > 0.7:
            recommendations.append("Consider caching or pre-computation to reduce latency")
        
        if resource_cost > 0.7:
            recommendations.append("Consider model compression or quantization")
        
        if accuracy_benefit < 0.6:
            recommendations.append("Consider fine-tuning or using a more powerful model")
        
        if complexity_cost > 0.7:
            recommendations.append("Consider using a simpler pipeline for easier maintenance")
        
        if not recommendations:
            recommendations.append("Pipeline is well-balanced for general use cases")
        
        return CostBenefitAnalysis(
            pipeline_type=pipeline_type,
            accuracy_benefit=accuracy_benefit,
            latency_cost=latency_cost,
            resource_cost=resource_cost,
            complexity_cost=complexity_cost,
            net_benefit=net_benefit,
            roi_score=roi_score,
            recommendations=recommendations,
        )

    def get_recommendation(
        self,
        profile: RequirementProfile,
    ) -> List[PipelineRecommendation]:
        """
        Get pipeline recommendations based on requirements.
        
        Args:
            profile: RequirementProfile with requirements
            
        Returns:
            List of PipelineRecommendation objects, sorted by score
        """
        recommendations = []
        
        default_weights = {
            "query_type": 0.25,
            "latency": 0.25,
            "accuracy": 0.30,
            "resource": 0.20,
        }
        weights = {**default_weights, **profile.custom_weights}
        
        for pipeline_type, metrics in self.scoreboard.get_all_metrics().items():
            pipeline_name = pipeline_type.value
            
            scores = {
                "query_type": self._get_query_type_score(pipeline_name, profile),
                "latency": self._get_latency_score(metrics, profile),
                "accuracy": self._get_accuracy_score(metrics, profile),
                "resource": self._get_resource_score(metrics, profile),
            }
            
            overall_score = sum(
                weights.get(k, 0.25) * v for k, v in scores.items()
            )
            
            meets_requirements = self._check_requirements_met(metrics, profile)
            
            strengths, weaknesses = self._get_strengths_weaknesses(
                pipeline_name,
                metrics,
                profile,
            )
            
            cost_benefit = self.calculate_cost_benefit(metrics)
            
            reasoning = self._generate_reasoning(
                pipeline_name,
                metrics,
                profile,
                scores,
            )
            
            recommendations.append(PipelineRecommendation(
                pipeline_type=pipeline_name,
                overall_score=overall_score,
                rank=0,
                meets_requirements=meets_requirements,
                strengths=strengths,
                weaknesses=weaknesses,
                cost_benefit_score=cost_benefit.net_benefit,
                detailed_scores=scores,
                reasoning=reasoning,
            ))
        
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        
        for i, rec in enumerate(recommendations, 1):
            rec.rank = i
        
        return recommendations

    def get_best_pipeline(
        self,
        profile: RequirementProfile,
    ) -> Optional[PipelineRecommendation]:
        """
        Get the best pipeline recommendation.
        
        Args:
            profile: RequirementProfile with requirements
            
        Returns:
            Best PipelineRecommendation, or None if no pipelines
        """
        recommendations = self.get_recommendation(profile)
        
        meeting_requirements = [r for r in recommendations if r.meets_requirements]
        if meeting_requirements:
            return meeting_requirements[0]
        
        if recommendations:
            return recommendations[0]
        
        return None

    def get_all_cost_benefit_analyses(self) -> Dict[str, CostBenefitAnalysis]:
        """
        Get cost-benefit analysis for all pipelines.
        
        Returns:
            Dictionary mapping pipeline names to CostBenefitAnalysis
        """
        analyses = {}
        for pipeline_type, metrics in self.scoreboard.get_all_metrics().items():
            analyses[pipeline_type.value] = self.calculate_cost_benefit(metrics)
        return analyses

    def export_recommendations(
        self,
        profile: RequirementProfile,
        filepath: Union[str, Path],
    ) -> None:
        """
        Export recommendations to a JSON file.
        
        Args:
            profile: RequirementProfile with requirements
            filepath: Path to save the JSON file
        """
        recommendations = self.get_recommendation(profile)
        cost_benefit = self.get_all_cost_benefit_analyses()
        
        data = {
            "profile": profile.to_dict(),
            "recommendations": [r.to_dict() for r in recommendations],
            "cost_benefit_analyses": {
                k: v.to_dict() for k, v in cost_benefit.items()
            },
            "best_pipeline": recommendations[0].pipeline_type if recommendations else None,
        }
        
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_recommendations_dict(
        self,
        profile: RequirementProfile,
    ) -> Dict[str, Any]:
        """
        Get recommendations as a dictionary.
        
        Args:
            profile: RequirementProfile with requirements
            
        Returns:
            Dictionary containing recommendations and analyses
        """
        recommendations = self.get_recommendation(profile)
        cost_benefit = self.get_all_cost_benefit_analyses()
        
        return {
            "profile": profile.to_dict(),
            "recommendations": [r.to_dict() for r in recommendations],
            "cost_benefit_analyses": {
                k: v.to_dict() for k, v in cost_benefit.items()
            },
            "best_pipeline": recommendations[0].pipeline_type if recommendations else None,
        }
