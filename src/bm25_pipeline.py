"""BM25 sparse retrieval evaluation pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

from src.bm25_retrieval import BM25Retriever
from src.synthetic_generator import SyntheticDocumentGenerator
from src.evaluation import RetrievalEvaluator
from src.scoreboard import Scoreboard


def generate_bm25_recommendations(
    aggregate_metrics: Dict,
    dataset_size: int
) -> List[str]:
    """Generate recommendations based on BM25 evaluation results.

    Args:
        aggregate_metrics: Dictionary of aggregate metrics.
        dataset_size: Number of documents in the dataset.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    mrr = aggregate_metrics.get("mean_mrr", 0)
    recall_10 = aggregate_metrics.get("mean_recall@10", 0)
    recall_100 = aggregate_metrics.get("mean_recall@100", 0)
    ndcg_10 = aggregate_metrics.get("mean_ndcg@10", 0)

    if mrr > 0.8:
        recommendations.append(
            f"Excellent MRR ({mrr:.3f}): BM25 performs very well on this dataset. "
            "The queries likely contain exact keyword matches with relevant documents."
        )
    elif mrr > 0.5:
        recommendations.append(
            f"Good MRR ({mrr:.3f}): BM25 provides reasonable baseline performance. "
            "Consider hybrid retrieval combining BM25 with dense embeddings for improvement."
        )
    else:
        recommendations.append(
            f"Low MRR ({mrr:.3f}): BM25 struggles with semantic matching. "
            "This suggests queries use different vocabulary than documents. "
            "Dense retrieval or query expansion may help significantly."
        )

    if recall_10 > 0.8:
        recommendations.append(
            f"Strong Recall@10 ({recall_10:.3f}): BM25 effectively retrieves relevant documents "
            "in the top 10 results. Good for re-ranking pipelines."
        )
    elif recall_10 < 0.5:
        recommendations.append(
            f"Low Recall@10 ({recall_10:.3f}): Consider increasing retrieval depth or "
            "using dense retrieval for better semantic coverage."
        )

    if recall_100 is not None and recall_100 > 0:
        if recall_100 > 0.9:
            recommendations.append(
                f"Excellent Recall@100 ({recall_100:.3f}): BM25 captures most relevant documents "
                "within top 100. Suitable as first-stage retriever in multi-stage pipelines."
            )

    recommendations.append(
        "BM25 Strengths: Exact keyword matching, no training required, interpretable scores, "
        "efficient for large corpora. Works well when queries contain document keywords."
    )

    recommendations.append(
        "BM25 Weaknesses: Cannot capture semantic similarity, struggles with paraphrases "
        "and synonyms, sensitive to vocabulary mismatch between queries and documents."
    )

    return recommendations


def generate_analysis(
    aggregate_metrics: Dict,
    per_query_results: List[Dict],
    dataset_info: Dict
) -> Dict[str, Any]:
    """Generate analysis of BM25 performance.

    Args:
        aggregate_metrics: Dictionary of aggregate metrics.
        per_query_results: List of per-query results.
        dataset_info: Information about the dataset.

    Returns:
        Dictionary containing analysis.
    """
    analysis = {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "comparison_notes": []
    }

    mrr = aggregate_metrics.get("mean_mrr", 0)
    ndcg_10 = aggregate_metrics.get("mean_ndcg@10", 0)
    recall_10 = aggregate_metrics.get("mean_recall@10", 0)

    if mrr > 0.7:
        analysis["summary"] = "BM25 performs well on this synthetic dataset, indicating strong keyword overlap between queries and documents."
    elif mrr > 0.4:
        analysis["summary"] = "BM25 shows moderate performance, suggesting partial vocabulary match between queries and documents."
    else:
        analysis["summary"] = "BM25 struggles on this dataset, likely due to semantic variations and vocabulary mismatch."

    analysis["strengths"] = [
        "No training or fine-tuning required",
        "Fast indexing and retrieval",
        "Interpretable term-based scoring",
        "Works well with exact keyword matches",
        "Memory efficient compared to dense retrieval"
    ]

    analysis["weaknesses"] = [
        "Cannot capture semantic similarity",
        "Struggles with synonyms and paraphrases",
        "Sensitive to query-document vocabulary mismatch",
        "No understanding of word context or meaning",
        "Performance degrades with short queries"
    ]

    analysis["comparison_notes"] = [
        "BM25 serves as a strong baseline for retrieval tasks",
        "Dense retrieval typically outperforms BM25 on semantic similarity tasks",
        "Hybrid approaches (BM25 + dense) often achieve best results",
        "BM25 is preferred when exact keyword matching is important"
    ]

    failed_queries = [r for r in per_query_results if r.get("mrr", 0) == 0]
    if failed_queries:
        analysis["failure_analysis"] = {
            "num_failed_queries": len(failed_queries),
            "failure_rate": len(failed_queries) / len(per_query_results) if per_query_results else 0,
            "note": "Queries with MRR=0 indicate no relevant document was retrieved"
        }

    return analysis


def run_bm25_pipeline(
    num_pairs: int = 100,
    top_k: int = 100,
    output_dir: str = "results/bm25",
    seed: Optional[int] = 42,
    k1: float = 1.5,
    b: float = 0.75
) -> Dict[str, Any]:
    """Run the complete BM25 retrieval evaluation pipeline.

    Args:
        num_pairs: Number of document-query pairs to generate.
        top_k: Number of results to retrieve per query.
        output_dir: Directory to save output files.
        seed: Random seed for reproducibility.
        k1: BM25 term frequency saturation parameter.
        b: BM25 document length normalization parameter.

    Returns:
        Dictionary with evaluation results.
    """
    print(f"Initializing BM25 retriever with k1={k1}, b={b}")
    retriever = BM25Retriever(k1=k1, b=b)

    print(f"Generating synthetic dataset with {num_pairs} pairs...")
    generator = SyntheticDocumentGenerator(seed=seed)
    dataset = generator.generate_dataset(
        num_pairs=num_pairs,
        include_hard_negatives=True
    )

    print(f"Generated {len(dataset.documents)} documents and {len(dataset.queries)} queries")

    print("Indexing documents...")
    retriever.index_documents(dataset.documents, dataset.document_ids)

    print(f"Running retrieval for {len(dataset.queries)} queries...")
    all_results = retriever.batch_search(dataset.queries, top_k=top_k)

    print("Evaluating retrieval results...")
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10, 100])

    for query_idx, (query, results) in enumerate(zip(dataset.queries, all_results)):
        query_key = f"q_{query_idx}"
        retrieved_ids = [doc_id for doc_id, _, _ in results]

        relevant_docs = set()
        relevance_scores = {}
        if query_key in dataset.relevance_map:
            for doc_id, rel in dataset.relevance_map[query_key].items():
                if rel > 0:
                    relevant_docs.add(doc_id)
                relevance_scores[doc_id] = rel

        evaluator.evaluate_query(
            query_id=query_key,
            retrieved=retrieved_ids,
            relevant=relevant_docs,
            relevance_scores=relevance_scores
        )

    aggregate_metrics = evaluator.get_aggregate_metrics()

    print("Generating scoreboard...")
    scoreboard = Scoreboard(experiment_name="bm25_baseline")

    scoreboard.add_per_query_results(evaluator.results)
    scoreboard.add_aggregate_results(aggregate_metrics)
    scoreboard.add_model_info(retriever.get_model_info())

    dataset_info = generator.get_statistics()
    dataset_info["num_documents"] = len(dataset.documents)
    dataset_info["num_queries"] = len(dataset.queries)
    scoreboard.add_dataset_info(dataset_info)

    recommendations = generate_bm25_recommendations(
        aggregate_metrics=aggregate_metrics,
        dataset_size=len(dataset.documents)
    )
    scoreboard.add_recommendations(recommendations)

    analysis = generate_analysis(
        aggregate_metrics=aggregate_metrics,
        per_query_results=evaluator.results,
        dataset_info=dataset_info
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_data = {
        "experiment_name": "bm25_baseline",
        "timestamp": datetime.now().isoformat(),
        "model_info": retriever.get_model_info(),
        "dataset_info": dataset_info,
        "aggregate_metrics": {
            "mrr": aggregate_metrics.get("mean_mrr", 0),
            "ndcg@10": aggregate_metrics.get("mean_ndcg@10", 0),
            "recall@10": aggregate_metrics.get("mean_recall@10", 0),
            "recall@100": aggregate_metrics.get("mean_recall@100", 0),
            "precision@1": aggregate_metrics.get("mean_precision@1", 0),
            "precision@10": aggregate_metrics.get("mean_precision@10", 0),
        },
        "full_aggregate_metrics": aggregate_metrics,
        "per_query_results": evaluator.results,
        "analysis": analysis,
        "recommendations": recommendations
    }

    json_path = output_path / "bm25_results.json"
    print(f"Saving results to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    scoreboard.print_summary()

    print("\n" + "=" * 60)
    print("KEY METRICS (for comparison with dense retrieval):")
    print("=" * 60)
    print(f"  MRR:        {aggregate_metrics.get('mean_mrr', 0):.4f}")
    print(f"  NDCG@10:    {aggregate_metrics.get('mean_ndcg@10', 0):.4f}")
    print(f"  Recall@10:  {aggregate_metrics.get('mean_recall@10', 0):.4f}")
    print(f"  Recall@100: {aggregate_metrics.get('mean_recall@100', 0):.4f}")
    print("=" * 60)

    return results_data


def main():
    """Main entry point for the BM25 pipeline."""
    parser = argparse.ArgumentParser(
        description="BM25 Sparse Retrieval Evaluation Pipeline"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of document-query pairs to generate"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bm25",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (term frequency saturation)"
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (document length normalization)"
    )

    args = parser.parse_args()

    try:
        run_bm25_pipeline(
            num_pairs=args.num_pairs,
            top_k=args.top_k,
            output_dir=args.output_dir,
            seed=args.seed,
            k1=args.k1,
            b=args.b
        )
        print("\nBM25 Pipeline completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError running pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
