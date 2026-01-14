"""
Command-line interface for the Retrieval Pipeline Comparison Dashboard.

Usage:
    python -m retrieval_comparison.cli --help
    python -m retrieval_comparison.cli compare --input metrics.json --output ./results
    python -m retrieval_comparison.cli recommend --input metrics.json --query-type semantic
    python -m retrieval_comparison.cli report --input metrics.json --output report.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .scoreboard import UnifiedScoreboard
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrieval Pipeline Comparison Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare pipeline metrics and generate visualizations",
    )
    compare_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with pipeline metrics",
    )
    compare_parser.add_argument(
        "--output", "-o",
        default="./comparison_output",
        help="Output directory for results",
    )
    compare_parser.add_argument(
        "--format",
        choices=["html", "png", "both"],
        default="html",
        help="Output format for visualizations",
    )
    
    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Get pipeline recommendations based on requirements",
    )
    recommend_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with pipeline metrics",
    )
    recommend_parser.add_argument(
        "--output", "-o",
        help="Output JSON file for recommendations",
    )
    recommend_parser.add_argument(
        "--query-type",
        choices=["keyword_heavy", "semantic", "mixed", "unknown"],
        default="mixed",
        help="Type of queries",
    )
    recommend_parser.add_argument(
        "--latency",
        choices=["real_time", "low_latency", "moderate", "batch"],
        default="moderate",
        help="Latency requirement",
    )
    recommend_parser.add_argument(
        "--accuracy",
        choices=["critical", "high", "moderate", "low"],
        default="high",
        help="Accuracy requirement",
    )
    recommend_parser.add_argument(
        "--resources",
        choices=["minimal", "limited", "moderate", "unlimited"],
        default="moderate",
        help="Resource constraint",
    )
    recommend_parser.add_argument(
        "--max-latency-ms",
        type=float,
        help="Maximum acceptable latency in ms",
    )
    recommend_parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Minimum acceptable accuracy (0-1)",
    )
    
    report_parser = subparsers.add_parser(
        "report",
        help="Generate a comprehensive comparison report",
    )
    report_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with pipeline metrics",
    )
    report_parser.add_argument(
        "--output", "-o",
        default="./report",
        help="Output path for the report",
    )
    report_parser.add_argument(
        "--format",
        choices=["html", "markdown", "json", "all"],
        default="html",
        help="Output format for the report",
    )
    report_parser.add_argument(
        "--query-type",
        choices=["keyword_heavy", "semantic", "mixed", "unknown"],
        default="mixed",
        help="Type of queries for recommendations",
    )
    
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a demonstration with sample data",
    )
    demo_parser.add_argument(
        "--output", "-o",
        default="./demo_output",
        help="Output directory for demo results",
    )
    
    return parser.parse_args()


def load_scoreboard(input_path: str) -> UnifiedScoreboard:
    """Load scoreboard from JSON file."""
    scoreboard = UnifiedScoreboard()
    scoreboard.load_from_json(input_path)
    return scoreboard


def create_profile(args: argparse.Namespace) -> RequirementProfile:
    """Create requirement profile from arguments."""
    return RequirementProfile(
        query_type=QueryType(getattr(args, "query_type", "mixed")),
        latency_requirement=LatencyRequirement(getattr(args, "latency", "moderate")),
        accuracy_requirement=AccuracyRequirement(getattr(args, "accuracy", "high")),
        resource_constraint=ResourceConstraint(getattr(args, "resources", "moderate")),
        max_latency_ms=getattr(args, "max_latency_ms", None),
        min_accuracy=getattr(args, "min_accuracy", None),
    )


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare command."""
    print(f"Loading metrics from {args.input}...")
    scoreboard = load_scoreboard(args.input)
    print(f"Loaded {len(scoreboard)} pipelines")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    viz = VisualizationDashboard(scoreboard)
    
    use_plotly = args.format in ["html", "both"]
    viz_paths = viz.create_comprehensive_dashboard(output_dir, use_plotly=use_plotly)
    
    if args.format == "both":
        viz.create_comprehensive_dashboard(output_dir / "png", use_plotly=False)
    
    print(f"\nSummary:")
    summary = scoreboard.get_summary_table()
    for pipeline, metrics in summary.items():
        print(f"  {pipeline.upper()}: "
              f"accuracy={metrics['accuracy_score']:.3f}, "
              f"latency={metrics['latency_ms']:.1f}ms")
    
    print(f"\nVisualizations saved to: {output_dir}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """Execute recommend command."""
    print(f"Loading metrics from {args.input}...")
    scoreboard = load_scoreboard(args.input)
    
    profile = create_profile(args)
    print(f"\nRequirement Profile:")
    print(f"  Query Type: {profile.query_type.value}")
    print(f"  Latency: {profile.latency_requirement.value}")
    print(f"  Accuracy: {profile.accuracy_requirement.value}")
    print(f"  Resources: {profile.resource_constraint.value}")
    
    engine = RecommendationEngine(scoreboard)
    recommendations = engine.get_recommendation(profile)
    
    print("\nRecommendations:")
    for rec in recommendations:
        status = "[MEETS REQUIREMENTS]" if rec.meets_requirements else "[DOES NOT MEET]"
        print(f"\n  {rec.rank}. {rec.pipeline_type.upper()} {status}")
        print(f"     Score: {rec.overall_score:.3f}")
        print(f"     {rec.reasoning}")
    
    best = engine.get_best_pipeline(profile)
    if best:
        print(f"\n*** Best Pipeline: {best.pipeline_type.upper()} ***")
    
    if args.output:
        engine.export_recommendations(profile, args.output)
        print(f"\nRecommendations saved to: {args.output}")
    
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Execute report command."""
    print(f"Loading metrics from {args.input}...")
    scoreboard = load_scoreboard(args.input)
    
    profile = create_profile(args)
    
    print("Generating report...")
    report = ReportGenerator(scoreboard, profile)
    
    output_path = Path(args.output)
    
    if args.format == "all":
        paths = report.save_all(output_path)
        print(f"\nReports saved:")
        for fmt, path in paths.items():
            if isinstance(path, dict):
                print(f"  {fmt}:")
                for name, p in path.items():
                    print(f"    - {name}: {p}")
            else:
                print(f"  {fmt}: {path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == "html":
            if not str(output_path).endswith(".html"):
                output_path = output_path.with_suffix(".html")
            report.save_html(output_path)
        elif args.format == "markdown":
            if not str(output_path).endswith(".md"):
                output_path = output_path.with_suffix(".md")
            report.save_markdown(output_path)
        elif args.format == "json":
            if not str(output_path).endswith(".json"):
                output_path = output_path.with_suffix(".json")
            report.save_json(output_path)
        
        print(f"\nReport saved to: {output_path}")
    
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Execute demo command."""
    from .example import run_example
    
    print("Running demonstration with sample data...")
    output_paths = run_example(args.output)
    
    print("\nDemo completed!")
    print(f"Output files:")
    for name, path in output_paths.items():
        if isinstance(path, dict):
            print(f"  {name}:")
            for subname, subpath in path.items():
                print(f"    - {subname}: {subpath}")
        else:
            print(f"  {name}: {path}")
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.command is None:
        print("Error: No command specified. Use --help for usage information.")
        return 1
    
    commands = {
        "compare": cmd_compare,
        "recommend": cmd_recommend,
        "report": cmd_report,
        "demo": cmd_demo,
    }
    
    cmd_func = commands.get(args.command)
    if cmd_func is None:
        print(f"Error: Unknown command '{args.command}'")
        return 1
    
    try:
        return cmd_func(args)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
