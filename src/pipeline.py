"""Main pipeline script for dense vector retrieval evaluation."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.dense_retrieval import DenseRetriever
from src.synthetic_generator import SyntheticDocumentGenerator
from src.evaluation import RetrievalEvaluator
from src.scoreboard import Scoreboard, generate_recommendations


def run_pipeline(
    model_name: str = "all-MiniLM-L6-v2",
    index_type: str = "flat",
    num_pairs: int = 100,
    top_k: int = 10,
    output_dir: str = "results",
    seed: Optional[int] = 42
) -> Scoreboard:
    """Run the complete dense retrieval evaluation pipeline.

    Args:
        model_name: Name of the sentence-transformer model to use.
        index_type: Type of FAISS index ('flat', 'ivf', 'hnsw').
        num_pairs: Number of document-query pairs to generate.
        top_k: Number of results to retrieve per query.
        output_dir: Directory to save output files.
        seed: Random seed for reproducibility.

    Returns:
        Scoreboard with evaluation results.
    """
    print(f"Initializing dense retriever with model: {model_name}")
    retriever = DenseRetriever(model_name=model_name, index_type=index_type)

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
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

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
    scoreboard = Scoreboard(experiment_name=f"dense_retrieval_{model_name}")

    scoreboard.add_per_query_results(evaluator.results)
    scoreboard.add_aggregate_results(aggregate_metrics)
    scoreboard.add_model_info(retriever.get_model_info())

    dataset_info = generator.get_statistics()
    dataset_info["num_documents"] = len(dataset.documents)
    dataset_info["num_queries"] = len(dataset.queries)
    scoreboard.add_dataset_info(dataset_info)

    recommendations = generate_recommendations(
        aggregate_metrics=aggregate_metrics,
        model_name=model_name,
        dataset_size=len(dataset.documents)
    )
    scoreboard.add_recommendations(recommendations)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "results.json"
    csv_path = output_path / "per_query_results.csv"
    aggregate_csv_path = output_path / "aggregate_results.csv"

    print(f"Saving results to {output_path}...")
    scoreboard.to_json(str(json_path))
    scoreboard.to_csv(str(csv_path))
    scoreboard.to_aggregate_csv(str(aggregate_csv_path))

    scoreboard.print_summary()

    return scoreboard


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Dense Vector Retrieval Evaluation Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type"
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
        default=10,
        help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            model_name=args.model,
            index_type=args.index_type,
            num_pairs=args.num_pairs,
            top_k=args.top_k,
            output_dir=args.output_dir,
            seed=args.seed
        )
        print("\nPipeline completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError running pipeline: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
