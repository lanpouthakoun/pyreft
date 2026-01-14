"""
Hybrid Retrieval Pipeline - Main Script

Demonstrates the hybrid retrieval system combining BM25 and dense vector search
with multiple fusion strategies.

Usage:
    python -m hybrid_retrieval.main

This script:
1. Generates synthetic documents and queries
2. Evaluates BM25, dense, and hybrid retrieval methods
3. Outputs a comparative scoreboard
4. Provides fusion weight tuning recommendations
"""

import argparse
import sys
from typing import Optional

from .evaluation import RetrievalEvaluator, run_evaluation_demo
from .synthetic_data import SyntheticDataGenerator
from .hybrid_retriever import HybridRetriever


def tune_fusion_weights(
    documents: list,
    queries: list,
    relevance_labels: list,
    alpha_range: tuple = (0.0, 1.0),
    num_steps: int = 11,
) -> dict:
    """
    Tune fusion weights by grid search.

    Args:
        documents: List of documents
        queries: List of queries
        relevance_labels: Relevance labels for each query
        alpha_range: Range of alpha values to search
        num_steps: Number of steps in the grid

    Returns:
        Dictionary with tuning results and recommendations
    """
    import numpy as np
    from .metrics import RetrievalMetrics

    retriever = HybridRetriever(
        fusion_strategy="weighted",
        documents=documents,
    )

    alpha_values = np.linspace(alpha_range[0], alpha_range[1], num_steps)
    results = []

    for alpha in alpha_values:
        retriever.set_fusion_strategy("weighted", {"alpha": alpha})

        ndcg_scores = []
        mrr_scores = []

        for query, relevance in zip(queries, relevance_labels):
            retrieved_results = retriever.retrieve(query, top_k=10)
            retrieved = [doc_idx for doc_idx, _, _ in retrieved_results]
            relevant = set(relevance.keys())

            ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance, 10)
            mrr = RetrievalMetrics.reciprocal_rank(retrieved, relevant)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

        results.append({
            "alpha": alpha,
            "ndcg@10": np.mean(ndcg_scores),
            "mrr": np.mean(mrr_scores),
        })

    best_ndcg = max(results, key=lambda x: x["ndcg@10"])
    best_mrr = max(results, key=lambda x: x["mrr"])

    recommendations = []
    recommendations.append(
        f"Best alpha for NDCG@10: {best_ndcg['alpha']:.2f} "
        f"(score: {best_ndcg['ndcg@10']:.4f})"
    )
    recommendations.append(
        f"Best alpha for MRR: {best_mrr['alpha']:.2f} "
        f"(score: {best_mrr['mrr']:.4f})"
    )

    if abs(best_ndcg["alpha"] - best_mrr["alpha"]) < 0.2:
        avg_alpha = (best_ndcg["alpha"] + best_mrr["alpha"]) / 2
        recommendations.append(
            f"Recommended alpha: {avg_alpha:.2f} (balanced for both metrics)"
        )
    else:
        recommendations.append(
            "NDCG and MRR prefer different weights. Choose based on your priority:"
        )
        recommendations.append(
            f"  - For ranking quality (NDCG): use alpha={best_ndcg['alpha']:.2f}"
        )
        recommendations.append(
            f"  - For first result quality (MRR): use alpha={best_mrr['alpha']:.2f}"
        )

    return {
        "grid_results": results,
        "best_ndcg": best_ndcg,
        "best_mrr": best_mrr,
        "recommendations": recommendations,
    }


def print_tuning_results(tuning_results: dict) -> str:
    """Format tuning results as a printable string."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("FUSION WEIGHT TUNING RESULTS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Grid Search Results (alpha = BM25 weight, 1-alpha = dense weight):")
    lines.append("-" * 60)
    lines.append(f"{'Alpha':<10}{'NDCG@10':<15}{'MRR':<15}")
    lines.append("-" * 60)

    for result in tuning_results["grid_results"]:
        lines.append(
            f"{result['alpha']:<10.2f}"
            f"{result['ndcg@10']:<15.4f}"
            f"{result['mrr']:<15.4f}"
        )

    lines.append("")
    lines.append("TUNING RECOMMENDATIONS")
    lines.append("-" * 60)

    for rec in tuning_results["recommendations"]:
        lines.append(f"  {rec}")

    lines.append("")
    lines.append("GENERAL FUSION WEIGHT GUIDELINES")
    lines.append("-" * 60)
    lines.append("  - alpha > 0.5: Favor BM25 (keyword matching)")
    lines.append("    Best for: exact term matching, technical queries, known-item search")
    lines.append("")
    lines.append("  - alpha < 0.5: Favor dense retrieval (semantic matching)")
    lines.append("    Best for: conceptual queries, paraphrased questions, semantic similarity")
    lines.append("")
    lines.append("  - alpha = 0.5: Balanced approach")
    lines.append("    Best for: general-purpose search, mixed query types")
    lines.append("")
    lines.append("  - RRF (k=60): No tuning required, robust default")
    lines.append("    Best for: when you don't have validation data for tuning")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main(
    num_documents: int = 200,
    num_queries: int = 100,
    seed: int = 42,
    tune_weights: bool = True,
) -> None:
    """
    Run the hybrid retrieval evaluation pipeline.

    Args:
        num_documents: Number of synthetic documents to generate
        num_queries: Number of synthetic queries to generate
        seed: Random seed for reproducibility
        tune_weights: Whether to run weight tuning
    """
    print("Hybrid Retrieval Pipeline")
    print("=" * 80)
    print(f"Generating {num_documents} documents and {num_queries} queries...")
    print("")

    generator = SyntheticDataGenerator(seed=seed)
    dataset = generator.generate(
        num_documents=num_documents,
        num_queries=num_queries,
    )

    print(f"Generated {len(dataset.documents)} documents")
    print(f"Generated {len(dataset.queries)} queries")
    print("")

    print("Running evaluation...")
    print("(This may take a few minutes for embedding computation)")
    print("")

    evaluator = RetrievalEvaluator()
    results = evaluator.evaluate_all_methods(
        documents=dataset.documents,
        queries=dataset.queries,
        relevance_labels=dataset.relevance_labels,
    )

    scoreboard = evaluator.generate_scoreboard(results)
    output = evaluator.print_scoreboard(scoreboard)
    print(output)

    if tune_weights:
        print("\nRunning fusion weight tuning...")
        tuning_results = tune_fusion_weights(
            documents=dataset.documents,
            queries=dataset.queries,
            relevance_labels=dataset.relevance_labels,
        )
        tuning_output = print_tuning_results(tuning_results)
        print(tuning_output)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval Pipeline Evaluation"
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=200,
        help="Number of synthetic documents to generate (default: 200)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of synthetic queries to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip fusion weight tuning",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        num_documents=args.num_documents,
        num_queries=args.num_queries,
        seed=args.seed,
        tune_weights=not args.no_tune,
    )
