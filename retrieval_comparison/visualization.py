"""
Visualization Dashboard Module

Creates charts and tables comparing pipeline performance using matplotlib and plotly.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import numpy as np

from .scoreboard import UnifiedScoreboard, PipelineMetrics, PipelineType


class VisualizationDashboard:
    """
    Visualization dashboard for comparing retrieval pipeline performance.
    
    Provides various chart types:
    - Bar charts for metric comparisons
    - Radar charts for multi-dimensional comparison
    - Latency vs accuracy trade-off plots
    - Resource usage comparisons
    - Heatmaps for correlation analysis
    """

    PIPELINE_COLORS = {
        "bm25": "#1f77b4",
        "dense": "#ff7f0e",
        "hybrid": "#2ca02c",
        "colbert": "#d62728",
    }

    def __init__(self, scoreboard: UnifiedScoreboard):
        """
        Initialize the visualization dashboard.
        
        Args:
            scoreboard: UnifiedScoreboard containing pipeline metrics
        """
        self.scoreboard = scoreboard
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if visualization libraries are available."""
        if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
            raise ImportError(
                "Neither matplotlib nor plotly is available. "
                "Please install at least one: pip install matplotlib plotly"
            )

    def _get_pipeline_color(self, pipeline_name: str) -> str:
        """Get color for a pipeline."""
        return self.PIPELINE_COLORS.get(pipeline_name.lower(), "#7f7f7f")

    def create_metric_bar_chart(
        self,
        metrics: List[str],
        title: str = "Pipeline Metric Comparison",
        use_plotly: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Figure, Any]:
        """
        Create a bar chart comparing specified metrics across pipelines.
        
        Args:
            metrics: List of metric names to compare
            title: Chart title
            use_plotly: If True, use plotly; otherwise use matplotlib
            save_path: Optional path to save the chart
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        summary = self.scoreboard.get_summary_table()
        pipelines = list(summary.keys())
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            for metric in metrics:
                values = [summary[p].get(metric, 0) for p in pipelines]
                fig.add_trace(go.Bar(
                    name=metric.replace("_", " ").title(),
                    x=pipelines,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Pipeline",
                yaxis_title="Score",
                barmode="group",
                legend_title="Metrics",
                template="plotly_white",
            )
            
            if save_path:
                fig.write_html(str(save_path))
            
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(pipelines))
            width = 0.8 / len(metrics)
            
            for i, metric in enumerate(metrics):
                values = [summary[p].get(metric, 0) for p in pipelines]
                offset = (i - len(metrics) / 2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title())
                
                for bar, val in zip(bars, values):
                    ax.annotate(
                        f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            
            ax.set_xlabel("Pipeline")
            ax.set_ylabel("Score")
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(pipelines)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            
            return fig
        
        raise ImportError("No visualization library available")

    def create_radar_chart(
        self,
        metrics: Optional[List[str]] = None,
        title: str = "Multi-dimensional Pipeline Comparison",
        use_plotly: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Figure, Any]:
        """
        Create a radar chart for multi-dimensional comparison.
        
        Args:
            metrics: List of metrics to include (default: accuracy metrics)
            title: Chart title
            use_plotly: If True, use plotly; otherwise use matplotlib
            save_path: Optional path to save the chart
            
        Returns:
            Figure object
        """
        if metrics is None:
            metrics = ["precision", "recall", "f1_score", "mrr", "ndcg"]
        
        summary = self.scoreboard.get_summary_table()
        pipelines = list(summary.keys())
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            for pipeline in pipelines:
                values = [summary[pipeline].get(m, 0) for m in metrics]
                values.append(values[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill="toself",
                    name=pipeline.upper(),
                    line_color=self._get_pipeline_color(pipeline),
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                    )
                ),
                showlegend=True,
                title=title,
                template="plotly_white",
            )
            
            if save_path:
                fig.write_html(str(save_path))
            
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for pipeline in pipelines:
                values = [summary[pipeline].get(m, 0) for m in metrics]
                values += values[:1]
                
                ax.plot(
                    angles,
                    values,
                    "o-",
                    linewidth=2,
                    label=pipeline.upper(),
                    color=self._get_pipeline_color(pipeline),
                )
                ax.fill(angles, values, alpha=0.25, color=self._get_pipeline_color(pipeline))
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax.set_title(title, pad=20)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            
            return fig
        
        raise ImportError("No visualization library available")

    def create_latency_accuracy_plot(
        self,
        accuracy_metric: str = "ndcg",
        title: str = "Latency vs Accuracy Trade-off",
        use_plotly: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Figure, Any]:
        """
        Create a scatter plot showing latency vs accuracy trade-off.
        
        Args:
            accuracy_metric: Metric to use for accuracy (default: ndcg)
            title: Chart title
            use_plotly: If True, use plotly; otherwise use matplotlib
            save_path: Optional path to save the chart
            
        Returns:
            Figure object
        """
        summary = self.scoreboard.get_summary_table()
        pipelines = list(summary.keys())
        
        latencies = [summary[p].get("latency_ms", 0) for p in pipelines]
        accuracies = [summary[p].get(accuracy_metric, 0) for p in pipelines]
        throughputs = [summary[p].get("throughput_qps", 1) for p in pipelines]
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            for i, pipeline in enumerate(pipelines):
                fig.add_trace(go.Scatter(
                    x=[latencies[i]],
                    y=[accuracies[i]],
                    mode="markers+text",
                    name=pipeline.upper(),
                    text=[pipeline.upper()],
                    textposition="top center",
                    marker=dict(
                        size=max(10, throughputs[i] / 10),
                        color=self._get_pipeline_color(pipeline),
                    ),
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Latency (ms)",
                yaxis_title=f"{accuracy_metric.upper()} Score",
                template="plotly_white",
                showlegend=True,
            )
            
            fig.add_annotation(
                text="Bubble size represents throughput (QPS)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                showarrow=False,
            )
            
            if save_path:
                fig.write_html(str(save_path))
            
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i, pipeline in enumerate(pipelines):
                size = max(100, throughputs[i] * 2)
                ax.scatter(
                    latencies[i],
                    accuracies[i],
                    s=size,
                    c=self._get_pipeline_color(pipeline),
                    label=pipeline.upper(),
                    alpha=0.7,
                )
                ax.annotate(
                    pipeline.upper(),
                    (latencies[i], accuracies[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
            
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel(f"{accuracy_metric.upper()} Score")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.text(
                0.5,
                -0.12,
                "Bubble size represents throughput (QPS)",
                transform=ax.transAxes,
                ha="center",
            )
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            
            return fig
        
        raise ImportError("No visualization library available")

    def create_resource_comparison(
        self,
        title: str = "Resource Usage Comparison",
        use_plotly: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Figure, Any]:
        """
        Create a chart comparing resource usage across pipelines.
        
        Args:
            title: Chart title
            use_plotly: If True, use plotly; otherwise use matplotlib
            save_path: Optional path to save the chart
            
        Returns:
            Figure object
        """
        summary = self.scoreboard.get_summary_table()
        pipelines = list(summary.keys())
        
        if use_plotly and PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Memory Usage (MB)",
                    "Latency (ms)",
                    "Throughput (QPS)",
                    "Efficiency Score",
                ),
            )
            
            colors = [self._get_pipeline_color(p) for p in pipelines]
            
            memory = [summary[p].get("memory_mb", 0) for p in pipelines]
            fig.add_trace(
                go.Bar(x=pipelines, y=memory, marker_color=colors, showlegend=False),
                row=1,
                col=1,
            )
            
            latency = [summary[p].get("latency_ms", 0) for p in pipelines]
            fig.add_trace(
                go.Bar(x=pipelines, y=latency, marker_color=colors, showlegend=False),
                row=1,
                col=2,
            )
            
            throughput = [summary[p].get("throughput_qps", 0) for p in pipelines]
            fig.add_trace(
                go.Bar(x=pipelines, y=throughput, marker_color=colors, showlegend=False),
                row=2,
                col=1,
            )
            
            efficiency = [summary[p].get("efficiency_score", 0) for p in pipelines]
            fig.add_trace(
                go.Bar(x=pipelines, y=efficiency, marker_color=colors, showlegend=False),
                row=2,
                col=2,
            )
            
            fig.update_layout(
                title_text=title,
                template="plotly_white",
                height=600,
            )
            
            if save_path:
                fig.write_html(str(save_path))
            
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            colors = [self._get_pipeline_color(p) for p in pipelines]
            
            memory = [summary[p].get("memory_mb", 0) for p in pipelines]
            axes[0, 0].bar(pipelines, memory, color=colors)
            axes[0, 0].set_title("Memory Usage (MB)")
            axes[0, 0].set_ylabel("MB")
            
            latency = [summary[p].get("latency_ms", 0) for p in pipelines]
            axes[0, 1].bar(pipelines, latency, color=colors)
            axes[0, 1].set_title("Latency (ms)")
            axes[0, 1].set_ylabel("ms")
            
            throughput = [summary[p].get("throughput_qps", 0) for p in pipelines]
            axes[1, 0].bar(pipelines, throughput, color=colors)
            axes[1, 0].set_title("Throughput (QPS)")
            axes[1, 0].set_ylabel("QPS")
            
            efficiency = [summary[p].get("efficiency_score", 0) for p in pipelines]
            axes[1, 1].bar(pipelines, efficiency, color=colors)
            axes[1, 1].set_title("Efficiency Score")
            axes[1, 1].set_ylabel("Score")
            
            for ax in axes.flat:
                ax.grid(axis="y", alpha=0.3)
            
            fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            
            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            
            return fig
        
        raise ImportError("No visualization library available")

    def create_ranking_table(
        self,
        metrics: Optional[List[str]] = None,
        use_plotly: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[Figure, Any]:
        """
        Create a table showing pipeline rankings across metrics.
        
        Args:
            metrics: List of metrics to include
            use_plotly: If True, use plotly; otherwise use matplotlib
            save_path: Optional path to save the table
            
        Returns:
            Figure object
        """
        if metrics is None:
            metrics = [
                "precision",
                "recall",
                "f1_score",
                "mrr",
                "ndcg",
                "latency_ms",
                "throughput_qps",
            ]
        
        summary = self.scoreboard.get_summary_table()
        pipelines = list(summary.keys())
        
        rankings = {}
        for metric in metrics:
            ascending = metric in ["latency_ms", "memory_mb"]
            ranking = self.scoreboard.get_ranking(metric, ascending=ascending)
            for rank, (pipeline, _) in enumerate(ranking, 1):
                if pipeline not in rankings:
                    rankings[pipeline] = {}
                rankings[pipeline][metric] = rank
        
        if use_plotly and PLOTLY_AVAILABLE:
            header_values = ["Pipeline"] + [m.replace("_", " ").title() for m in metrics]
            cell_values = [[p.upper() for p in pipelines]]
            
            for metric in metrics:
                cell_values.append([rankings[p].get(metric, "-") for p in pipelines])
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=header_values,
                    fill_color="paleturquoise",
                    align="center",
                    font=dict(size=12),
                ),
                cells=dict(
                    values=cell_values,
                    fill_color="lavender",
                    align="center",
                    font=dict(size=11),
                ),
            )])
            
            fig.update_layout(
                title="Pipeline Rankings (1 = Best)",
                template="plotly_white",
            )
            
            if save_path:
                fig.write_html(str(save_path))
            
            return fig
        
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.axis("off")
            
            header = ["Pipeline"] + [m.replace("_", " ").title() for m in metrics]
            cell_text = []
            
            for pipeline in pipelines:
                row = [pipeline.upper()]
                for metric in metrics:
                    row.append(str(rankings[pipeline].get(metric, "-")))
                cell_text.append(row)
            
            table = ax.table(
                cellText=cell_text,
                colLabels=header,
                cellLoc="center",
                loc="center",
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            for i in range(len(header)):
                table[(0, i)].set_facecolor("paleturquoise")
            
            ax.set_title("Pipeline Rankings (1 = Best)", pad=20)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            
            return fig
        
        raise ImportError("No visualization library available")

    def create_comprehensive_dashboard(
        self,
        output_dir: Union[str, Path],
        use_plotly: bool = True,
    ) -> Dict[str, str]:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save all visualizations
            use_plotly: If True, use plotly; otherwise use matplotlib
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ext = ".html" if use_plotly and PLOTLY_AVAILABLE else ".png"
        
        charts = {}
        
        accuracy_metrics = ["precision", "recall", "f1_score", "mrr", "ndcg"]
        path = output_dir / f"accuracy_comparison{ext}"
        self.create_metric_bar_chart(
            metrics=accuracy_metrics,
            title="Accuracy Metrics Comparison",
            use_plotly=use_plotly,
            save_path=path,
        )
        charts["accuracy_comparison"] = str(path)
        
        path = output_dir / f"radar_chart{ext}"
        self.create_radar_chart(
            use_plotly=use_plotly,
            save_path=path,
        )
        charts["radar_chart"] = str(path)
        
        path = output_dir / f"latency_accuracy{ext}"
        self.create_latency_accuracy_plot(
            use_plotly=use_plotly,
            save_path=path,
        )
        charts["latency_accuracy"] = str(path)
        
        path = output_dir / f"resource_comparison{ext}"
        self.create_resource_comparison(
            use_plotly=use_plotly,
            save_path=path,
        )
        charts["resource_comparison"] = str(path)
        
        path = output_dir / f"ranking_table{ext}"
        self.create_ranking_table(
            use_plotly=use_plotly,
            save_path=path,
        )
        charts["ranking_table"] = str(path)
        
        return charts

    def get_dashboard_html(self) -> str:
        """
        Generate an HTML dashboard combining all visualizations.
        
        Returns:
            HTML string containing the dashboard
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for HTML dashboard generation")
        
        accuracy_metrics = ["precision", "recall", "f1_score", "mrr", "ndcg"]
        bar_fig = self.create_metric_bar_chart(metrics=accuracy_metrics, use_plotly=True)
        radar_fig = self.create_radar_chart(use_plotly=True)
        scatter_fig = self.create_latency_accuracy_plot(use_plotly=True)
        resource_fig = self.create_resource_comparison(use_plotly=True)
        table_fig = self.create_ranking_table(use_plotly=True)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Retrieval Pipeline Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-title {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .row {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .col {{
            flex: 1;
        }}
    </style>
</head>
<body>
    <h1 class="dashboard-title">Retrieval Pipeline Comparison Dashboard</h1>
    
    <div class="row">
        <div class="col chart-container">
            {bar_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
    
    <div class="row">
        <div class="col chart-container">
            {radar_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
        <div class="col chart-container">
            {scatter_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
    
    <div class="row">
        <div class="col chart-container">
            {resource_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
    
    <div class="row">
        <div class="col chart-container">
            {table_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
</body>
</html>
"""
        return html_content

    def save_dashboard_html(self, filepath: Union[str, Path]) -> None:
        """
        Save the HTML dashboard to a file.
        
        Args:
            filepath: Path to save the HTML file
        """
        html_content = self.get_dashboard_html()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(html_content)
