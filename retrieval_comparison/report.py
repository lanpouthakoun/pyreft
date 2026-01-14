"""
Report Generator Module

Generates comprehensive summary reports with findings and recommendations
for retrieval pipeline comparison.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json

from .scoreboard import UnifiedScoreboard, PipelineMetrics
from .recommendation import (
    RecommendationEngine,
    RequirementProfile,
    PipelineRecommendation,
    CostBenefitAnalysis,
)
from .visualization import VisualizationDashboard


@dataclass
class ReportSection:
    """A section of the report."""
    title: str
    content: str
    subsections: List["ReportSection"] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
        }


class ReportGenerator:
    """
    Generates comprehensive reports for retrieval pipeline comparison.
    
    Includes:
    - Executive summary
    - Detailed metrics comparison
    - Visualizations
    - Recommendations
    - Cost-benefit analysis
    - Actionable insights
    """

    def __init__(
        self,
        scoreboard: UnifiedScoreboard,
        profile: Optional[RequirementProfile] = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            scoreboard: UnifiedScoreboard containing pipeline metrics
            profile: Optional RequirementProfile for recommendations
        """
        self.scoreboard = scoreboard
        self.profile = profile or RequirementProfile()
        self.recommendation_engine = RecommendationEngine(scoreboard)
        self.visualization = VisualizationDashboard(scoreboard)
        self._report_sections: List[ReportSection] = []

    def _generate_executive_summary(self) -> ReportSection:
        """Generate executive summary section."""
        summary = self.scoreboard.get_summary_table()
        pipelines = list(summary.keys())
        
        recommendations = self.recommendation_engine.get_recommendation(self.profile)
        best = recommendations[0] if recommendations else None
        
        accuracy_ranking = self.scoreboard.get_accuracy_ranking()
        efficiency_ranking = self.scoreboard.get_efficiency_ranking()
        
        content_parts = [
            f"This report analyzes {len(pipelines)} retrieval pipelines: "
            f"{', '.join(p.upper() for p in pipelines)}.",
            "",
        ]
        
        if best:
            content_parts.extend([
                f"**Recommended Pipeline:** {best.pipeline_type.upper()}",
                f"- Overall Score: {best.overall_score:.2f}",
                f"- Meets Requirements: {'Yes' if best.meets_requirements else 'No'}",
                f"- {best.reasoning}",
                "",
            ])
        
        if accuracy_ranking:
            content_parts.extend([
                "**Accuracy Ranking:**",
                *[f"  {i+1}. {p.upper()} ({s:.3f})" for i, (p, s) in enumerate(accuracy_ranking)],
                "",
            ])
        
        if efficiency_ranking:
            content_parts.extend([
                "**Efficiency Ranking:**",
                *[f"  {i+1}. {p.upper()} ({s:.3f})" for i, (p, s) in enumerate(efficiency_ranking)],
            ])
        
        return ReportSection(
            title="Executive Summary",
            content="\n".join(content_parts),
        )

    def _generate_metrics_comparison(self) -> ReportSection:
        """Generate detailed metrics comparison section."""
        summary = self.scoreboard.get_summary_table()
        
        accuracy_content = ["**Accuracy Metrics:**", ""]
        accuracy_metrics = ["precision", "recall", "f1_score", "mrr", "ndcg", "map_score"]
        
        header = "| Pipeline | " + " | ".join(m.replace("_", " ").title() for m in accuracy_metrics) + " |"
        separator = "|" + "|".join(["---"] * (len(accuracy_metrics) + 1)) + "|"
        accuracy_content.extend([header, separator])
        
        for pipeline, metrics in summary.items():
            row = f"| {pipeline.upper()} | "
            row += " | ".join(f"{metrics.get(m, 0):.3f}" for m in accuracy_metrics)
            row += " |"
            accuracy_content.append(row)
        
        accuracy_section = ReportSection(
            title="Accuracy Metrics",
            content="\n".join(accuracy_content),
        )
        
        perf_content = ["**Performance Metrics:**", ""]
        perf_metrics = ["latency_ms", "throughput_qps", "memory_mb"]
        
        header = "| Pipeline | Latency (ms) | Throughput (QPS) | Memory (MB) |"
        separator = "|---|---|---|---|"
        perf_content.extend([header, separator])
        
        for pipeline, metrics in summary.items():
            row = f"| {pipeline.upper()} | "
            row += f"{metrics.get('latency_ms', 0):.1f} | "
            row += f"{metrics.get('throughput_qps', 0):.1f} | "
            row += f"{metrics.get('memory_mb', 0):.1f} |"
            perf_content.append(row)
        
        perf_section = ReportSection(
            title="Performance Metrics",
            content="\n".join(perf_content),
        )
        
        return ReportSection(
            title="Detailed Metrics Comparison",
            content="This section provides a detailed comparison of all metrics across pipelines.",
            subsections=[accuracy_section, perf_section],
        )

    def _generate_recommendations_section(self) -> ReportSection:
        """Generate recommendations section."""
        recommendations = self.recommendation_engine.get_recommendation(self.profile)
        
        content_parts = [
            f"Based on the requirement profile:",
            f"- Query Type: {self.profile.query_type.value}",
            f"- Latency Requirement: {self.profile.latency_requirement.value}",
            f"- Accuracy Requirement: {self.profile.accuracy_requirement.value}",
            f"- Resource Constraint: {self.profile.resource_constraint.value}",
            "",
            "**Pipeline Rankings:**",
            "",
        ]
        
        for rec in recommendations:
            content_parts.extend([
                f"### {rec.rank}. {rec.pipeline_type.upper()}",
                f"- Overall Score: {rec.overall_score:.3f}",
                f"- Meets Requirements: {'Yes' if rec.meets_requirements else 'No'}",
                f"- Cost-Benefit Score: {rec.cost_benefit_score:.3f}",
                "",
                "**Strengths:**",
                *[f"  - {s}" for s in rec.strengths[:3]],
                "",
                "**Weaknesses:**",
                *[f"  - {w}" for w in rec.weaknesses[:3]],
                "",
                f"**Reasoning:** {rec.reasoning}",
                "",
                "---",
                "",
            ])
        
        return ReportSection(
            title="Recommendations",
            content="\n".join(content_parts),
        )

    def _generate_cost_benefit_section(self) -> ReportSection:
        """Generate cost-benefit analysis section."""
        analyses = self.recommendation_engine.get_all_cost_benefit_analyses()
        
        content_parts = [
            "This section provides a cost-benefit analysis for each pipeline.",
            "",
            "| Pipeline | Accuracy Benefit | Latency Cost | Resource Cost | Net Benefit | ROI |",
            "|---|---|---|---|---|---|",
        ]
        
        for pipeline, analysis in analyses.items():
            row = f"| {pipeline.upper()} | "
            row += f"{analysis.accuracy_benefit:.3f} | "
            row += f"{analysis.latency_cost:.3f} | "
            row += f"{analysis.resource_cost:.3f} | "
            row += f"{analysis.net_benefit:.3f} | "
            row += f"{analysis.roi_score:.2f} |"
            content_parts.append(row)
        
        content_parts.extend(["", "**Detailed Analysis:**", ""])
        
        for pipeline, analysis in analyses.items():
            content_parts.extend([
                f"### {pipeline.upper()}",
                "",
                f"- **Accuracy Benefit:** {analysis.accuracy_benefit:.3f}",
                f"- **Latency Cost:** {analysis.latency_cost:.3f}",
                f"- **Resource Cost:** {analysis.resource_cost:.3f}",
                f"- **Complexity Cost:** {analysis.complexity_cost:.3f}",
                f"- **Net Benefit:** {analysis.net_benefit:.3f}",
                f"- **ROI Score:** {analysis.roi_score:.2f}",
                "",
                "**Recommendations:**",
                *[f"  - {r}" for r in analysis.recommendations],
                "",
            ])
        
        return ReportSection(
            title="Cost-Benefit Analysis",
            content="\n".join(content_parts),
        )

    def _generate_insights_section(self) -> ReportSection:
        """Generate actionable insights section."""
        summary = self.scoreboard.get_summary_table()
        recommendations = self.recommendation_engine.get_recommendation(self.profile)
        
        insights = []
        
        best_accuracy = max(summary.items(), key=lambda x: x[1].get("accuracy_score", 0))
        insights.append(
            f"**Highest Accuracy:** {best_accuracy[0].upper()} achieves the highest "
            f"accuracy score of {best_accuracy[1].get('accuracy_score', 0):.3f}."
        )
        
        best_latency = min(summary.items(), key=lambda x: x[1].get("latency_ms", float("inf")))
        insights.append(
            f"**Lowest Latency:** {best_latency[0].upper()} has the lowest latency "
            f"at {best_latency[1].get('latency_ms', 0):.1f}ms."
        )
        
        best_efficiency = max(summary.items(), key=lambda x: x[1].get("efficiency_score", 0))
        insights.append(
            f"**Best Efficiency:** {best_efficiency[0].upper()} offers the best "
            f"efficiency score of {best_efficiency[1].get('efficiency_score', 0):.3f}."
        )
        
        if recommendations:
            best = recommendations[0]
            insights.append(
                f"**Overall Recommendation:** For the given requirements, "
                f"{best.pipeline_type.upper()} is recommended with an overall "
                f"score of {best.overall_score:.3f}."
            )
        
        use_case_insights = [
            "",
            "**Use Case Recommendations:**",
            "",
            "- **Real-time applications:** Consider BM25 or Hybrid for low latency",
            "- **Semantic search:** Dense or ColBERT for best semantic understanding",
            "- **Resource-constrained:** BM25 for minimal resource usage",
            "- **Maximum accuracy:** ColBERT for state-of-the-art results",
            "- **Balanced needs:** Hybrid for best overall trade-off",
        ]
        
        content = "\n\n".join(insights) + "\n" + "\n".join(use_case_insights)
        
        return ReportSection(
            title="Actionable Insights",
            content=content,
        )

    def generate_report(self) -> List[ReportSection]:
        """
        Generate the complete report.
        
        Returns:
            List of ReportSection objects
        """
        self._report_sections = [
            self._generate_executive_summary(),
            self._generate_metrics_comparison(),
            self._generate_recommendations_section(),
            self._generate_cost_benefit_section(),
            self._generate_insights_section(),
        ]
        
        return self._report_sections

    def _section_to_markdown(self, section: ReportSection, level: int = 1) -> str:
        """Convert a section to markdown format."""
        lines = [
            "#" * level + " " + section.title,
            "",
            section.content,
            "",
        ]
        
        for subsection in section.subsections:
            lines.append(self._section_to_markdown(subsection, level + 1))
        
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """
        Convert the report to markdown format.
        
        Returns:
            Markdown string
        """
        if not self._report_sections:
            self.generate_report()
        
        header = [
            "# Retrieval Pipeline Comparison Report",
            "",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---",
            "",
        ]
        
        sections = [
            self._section_to_markdown(section)
            for section in self._report_sections
        ]
        
        return "\n".join(header) + "\n".join(sections)

    def to_html(self, include_visualizations: bool = True) -> str:
        """
        Convert the report to HTML format.
        
        Args:
            include_visualizations: Whether to include interactive visualizations
            
        Returns:
            HTML string
        """
        if not self._report_sections:
            self.generate_report()
        
        try:
            import markdown
            md_content = self.to_markdown()
            html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
        except ImportError:
            html_body = f"<pre>{self.to_markdown()}</pre>"
        
        viz_html = ""
        if include_visualizations:
            try:
                viz_html = self.visualization.get_dashboard_html()
                viz_html = viz_html.replace("<!DOCTYPE html>", "")
                viz_html = viz_html.replace("<html>", "")
                viz_html = viz_html.replace("</html>", "")
                viz_html = viz_html.replace("<head>", "")
                viz_html = viz_html.replace("</head>", "")
                viz_html = viz_html.replace("<body>", "")
                viz_html = viz_html.replace("</body>", "")
            except Exception:
                viz_html = ""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Retrieval Pipeline Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .report-container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .visualization-container {{
            margin-top: 40px;
        }}
        .timestamp {{
            color: #95a5a6;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ecf0f1;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        {html_body}
    </div>
    
    <div class="visualization-container">
        <h2>Interactive Visualizations</h2>
        {viz_html}
    </div>
</body>
</html>
"""
        return html

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report to a dictionary.
        
        Returns:
            Dictionary containing the report data
        """
        if not self._report_sections:
            self.generate_report()
        
        recommendations = self.recommendation_engine.get_recommendations_dict(self.profile)
        
        return {
            "generated_at": datetime.now().isoformat(),
            "profile": self.profile.to_dict(),
            "scoreboard": self.scoreboard.export_to_dict(),
            "recommendations": recommendations,
            "sections": [s.to_dict() for s in self._report_sections],
        }

    def save_markdown(self, filepath: Union[str, Path]) -> None:
        """
        Save the report as a markdown file.
        
        Args:
            filepath: Path to save the file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(self.to_markdown())

    def save_html(
        self,
        filepath: Union[str, Path],
        include_visualizations: bool = True,
    ) -> None:
        """
        Save the report as an HTML file.
        
        Args:
            filepath: Path to save the file
            include_visualizations: Whether to include visualizations
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(self.to_html(include_visualizations))

    def save_json(self, filepath: Union[str, Path]) -> None:
        """
        Save the report as a JSON file.
        
        Args:
            filepath: Path to save the file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_all(
        self,
        output_dir: Union[str, Path],
        base_name: str = "pipeline_comparison_report",
    ) -> Dict[str, str]:
        """
        Save the report in all formats.
        
        Args:
            output_dir: Directory to save files
            base_name: Base name for the files
            
        Returns:
            Dictionary mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        md_path = output_dir / f"{base_name}.md"
        self.save_markdown(md_path)
        paths["markdown"] = str(md_path)
        
        html_path = output_dir / f"{base_name}.html"
        self.save_html(html_path)
        paths["html"] = str(html_path)
        
        json_path = output_dir / f"{base_name}.json"
        self.save_json(json_path)
        paths["json"] = str(json_path)
        
        viz_paths = self.visualization.create_comprehensive_dashboard(
            output_dir / "visualizations"
        )
        paths["visualizations"] = viz_paths
        
        return paths
