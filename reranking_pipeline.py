"""
Two-Stage Retrieval Pipeline with Cross-Encoder Reranking

This module implements a two-stage retrieval pipeline that combines first-stage
retrieval (BM25 or dense) with cross-encoder reranking for improved accuracy.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    doc_id: str
    text: str
    score: float
    rank: int


@dataclass
class QueryResult:
    """Represents results for a single query."""
    query_id: str
    query_text: str
    results: List[RetrievalResult]
    retrieval_time_ms: float = 0.0
    reranking_time_ms: float = 0.0


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for a retrieval run."""
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    recall_at_10: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_reranking_time_ms: float = 0.0
    total_queries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "recall@10": self.recall_at_10,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "avg_reranking_time_ms": self.avg_reranking_time_ms,
            "total_queries": self.total_queries
        }


class FirstStageRetriever(ABC):
    """Abstract base class for first-stage retrievers."""
    
    @abstractmethod
    def index(self, documents: List[Dict[str, str]]) -> None:
        """Index a collection of documents."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[RetrievalResult], float]:
        """Retrieve top-k documents for a query. Returns results and time in ms."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the retriever."""
        pass


class BM25Retriever(FirstStageRetriever):
    """BM25-based first-stage retriever."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents: List[Dict[str, str]] = []
        self.tokenized_corpus: List[List[str]] = []
    
    @property
    def name(self) -> str:
        return "BM25"
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def index(self, documents: List[Dict[str, str]]) -> None:
        """Index documents using BM25."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank_bm25")
        
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
    
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[RetrievalResult], float]:
        """Retrieve top-k documents using BM25."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() first.")
        
        start_time = time.perf_counter()
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                doc_id=self.documents[idx].get("id", str(idx)),
                text=self.documents[idx]["text"],
                score=float(scores[idx]),
                rank=rank + 1
            ))
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return results, elapsed_ms


class DenseRetriever(FirstStageRetriever):
    """Dense retrieval using sentence-transformers bi-encoder."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.documents: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None
    
    @property
    def name(self) -> str:
        return f"Dense({self.model_name.split('/')[-1]})"
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            self.model = SentenceTransformer(self.model_name)
    
    def index(self, documents: List[Dict[str, str]]) -> None:
        """Index documents by computing embeddings."""
        self._load_model()
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[RetrievalResult], float]:
        """Retrieve top-k documents using cosine similarity."""
        if self.embeddings is None:
            raise ValueError("Index not built. Call index() first.")
        
        self._load_model()
        start_time = time.perf_counter()
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                doc_id=self.documents[idx].get("id", str(idx)),
                text=self.documents[idx]["text"],
                score=float(similarities[idx]),
                rank=rank + 1
            ))
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return results, elapsed_ms


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
    
    @property
    def name(self) -> str:
        return f"CrossEncoder({self.model_name.split('/')[-1]})"
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
            self.model = CrossEncoder(self.model_name)
    
    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 10) -> Tuple[List[RetrievalResult], float]:
        """Rerank candidates using cross-encoder scores."""
        if not candidates:
            return [], 0.0
        
        self._load_model()
        start_time = time.perf_counter()
        
        pairs = [[query, candidate.text] for candidate in candidates]
        scores = self.model.predict(pairs)
        
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        reranked_results = []
        for rank, (candidate, score) in enumerate(scored_candidates[:top_k]):
            reranked_results.append(RetrievalResult(
                doc_id=candidate.doc_id,
                text=candidate.text,
                score=float(score),
                rank=rank + 1
            ))
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return reranked_results, elapsed_ms


class TwoStageRetrievalPipeline:
    """Two-stage retrieval pipeline with optional reranking."""
    
    def __init__(
        self,
        first_stage_retriever: FirstStageRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        first_stage_k: int = 100,
        final_k: int = 10
    ):
        self.first_stage_retriever = first_stage_retriever
        self.reranker = reranker
        self.first_stage_k = first_stage_k
        self.final_k = final_k
    
    @property
    def name(self) -> str:
        base_name = self.first_stage_retriever.name
        if self.reranker:
            return f"{base_name} + {self.reranker.name}"
        return base_name
    
    def index(self, documents: List[Dict[str, str]]) -> None:
        """Index documents in the first-stage retriever."""
        self.first_stage_retriever.index(documents)
    
    def retrieve(self, query: str, query_id: str = "") -> QueryResult:
        """Retrieve and optionally rerank documents for a query."""
        first_stage_results, retrieval_time = self.first_stage_retriever.retrieve(
            query, top_k=self.first_stage_k
        )
        
        reranking_time = 0.0
        if self.reranker and first_stage_results:
            final_results, reranking_time = self.reranker.rerank(
                query, first_stage_results, top_k=self.final_k
            )
        else:
            final_results = first_stage_results[:self.final_k]
        
        return QueryResult(
            query_id=query_id,
            query_text=query,
            results=final_results,
            retrieval_time_ms=retrieval_time,
            reranking_time_ms=reranking_time
        )


class RetrievalEvaluator:
    """Evaluator for retrieval pipelines."""
    
    @staticmethod
    def compute_mrr(results: List[RetrievalResult], relevant_doc_ids: set) -> float:
        """Compute Mean Reciprocal Rank."""
        for result in results:
            if result.doc_id in relevant_doc_ids:
                return 1.0 / result.rank
        return 0.0
    
    @staticmethod
    def compute_dcg(relevances: List[float], k: int) -> float:
        """Compute Discounted Cumulative Gain at k."""
        relevances = relevances[:k]
        if not relevances:
            return 0.0
        dcg = relevances[0]
        for i, rel in enumerate(relevances[1:], start=2):
            dcg += rel / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def compute_ndcg(results: List[RetrievalResult], relevant_doc_ids: set, k: int = 10) -> float:
        """Compute Normalized Discounted Cumulative Gain at k."""
        relevances = [1.0 if r.doc_id in relevant_doc_ids else 0.0 for r in results[:k]]
        dcg = RetrievalEvaluator.compute_dcg(relevances, k)
        
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = RetrievalEvaluator.compute_dcg(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def compute_recall(results: List[RetrievalResult], relevant_doc_ids: set, k: int = 10) -> float:
        """Compute Recall at k."""
        if not relevant_doc_ids:
            return 0.0
        retrieved_ids = {r.doc_id for r in results[:k]}
        hits = len(retrieved_ids & relevant_doc_ids)
        return hits / len(relevant_doc_ids)
    
    def evaluate(
        self,
        pipeline: TwoStageRetrievalPipeline,
        queries: List[Dict[str, Any]],
        qrels: Dict[str, set]
    ) -> EvaluationMetrics:
        """Evaluate a pipeline on a set of queries with relevance judgments."""
        mrr_scores = []
        ndcg_scores = []
        recall_scores = []
        retrieval_times = []
        reranking_times = []
        
        for query_data in queries:
            query_id = query_data["id"]
            query_text = query_data["text"]
            relevant_docs = qrels.get(query_id, set())
            
            result = pipeline.retrieve(query_text, query_id)
            
            mrr_scores.append(self.compute_mrr(result.results, relevant_docs))
            ndcg_scores.append(self.compute_ndcg(result.results, relevant_docs, k=10))
            recall_scores.append(self.compute_recall(result.results, relevant_docs, k=10))
            retrieval_times.append(result.retrieval_time_ms)
            reranking_times.append(result.reranking_time_ms)
        
        return EvaluationMetrics(
            mrr=float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            ndcg_at_10=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            recall_at_10=float(np.mean(recall_scores)) if recall_scores else 0.0,
            avg_retrieval_time_ms=float(np.mean(retrieval_times)) if retrieval_times else 0.0,
            avg_reranking_time_ms=float(np.mean(reranking_times)) if reranking_times else 0.0,
            total_queries=len(queries)
        )


def load_msmarco_sample(num_queries: int = 100, num_docs: int = 10000) -> Tuple[List[Dict], List[Dict], Dict[str, set]]:
    """Load a sample from MS MARCO dataset for evaluation."""
    print(f"Loading MS MARCO sample (queries={num_queries}, docs={num_docs})...")
    
    try:
        corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
        queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries", trust_remote_code=True)
        qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading MS MARCO from BeIR: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_dataset(num_queries, num_docs)
    
    qrels: Dict[str, set] = {}
    for item in qrels_ds:
        qid = str(item["query-id"])
        did = str(item["corpus-id"])
        if qid not in qrels:
            qrels[qid] = set()
        qrels[qid].add(did)
    
    valid_query_ids = set(qrels.keys())
    queries = []
    for item in queries_ds:
        qid = str(item["_id"])
        if qid in valid_query_ids and len(queries) < num_queries:
            queries.append({"id": qid, "text": item["text"]})
    
    relevant_doc_ids = set()
    for qid in [q["id"] for q in queries]:
        relevant_doc_ids.update(qrels.get(qid, set()))
    
    documents = []
    doc_count = 0
    for item in corpus:
        did = str(item["_id"])
        if did in relevant_doc_ids or doc_count < num_docs:
            documents.append({
                "id": did,
                "text": item["text"] if item["text"] else item.get("title", "")
            })
            doc_count += 1
            if doc_count >= num_docs and len(relevant_doc_ids - {d["id"] for d in documents}) == 0:
                break
    
    filtered_qrels = {qid: qrels[qid] for qid in [q["id"] for q in queries] if qid in qrels}
    
    print(f"Loaded {len(documents)} documents, {len(queries)} queries")
    return documents, queries, filtered_qrels


def create_synthetic_dataset(num_queries: int = 100, num_docs: int = 1000) -> Tuple[List[Dict], List[Dict], Dict[str, set]]:
    """Create a synthetic dataset for testing when real data is unavailable."""
    print("Creating synthetic dataset for testing...")
    
    topics = [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "reinforcement learning", "neural networks",
        "data science", "artificial intelligence", "robotics", "automation"
    ]
    
    documents = []
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        documents.append({
            "id": f"doc_{i}",
            "text": f"This document discusses {topic}. It covers various aspects of {topic} "
                   f"including theory, applications, and recent advances in the field. "
                   f"Document number {i} provides comprehensive information about {topic}."
        })
    
    queries = []
    qrels: Dict[str, set] = {}
    for i in range(num_queries):
        topic = topics[i % len(topics)]
        qid = f"query_{i}"
        queries.append({
            "id": qid,
            "text": f"What are the latest developments in {topic}?"
        })
        relevant_docs = {f"doc_{j}" for j in range(num_docs) if j % len(topics) == i % len(topics)}
        qrels[qid] = relevant_docs
    
    return documents, queries, qrels


def run_experiments(
    num_queries: int = 50,
    num_docs: int = 5000,
    first_stage_k_values: List[int] = [20, 50, 100, 200],
    output_file: str = "reranking_results.json"
) -> Dict[str, Any]:
    """Run comprehensive experiments comparing different retrieval configurations."""
    
    print("=" * 60)
    print("Two-Stage Retrieval Pipeline Experiments")
    print("=" * 60)
    
    documents, queries, qrels = load_msmarco_sample(num_queries, num_docs)
    
    bm25_retriever = BM25Retriever()
    dense_retriever = DenseRetriever()
    cross_encoder = CrossEncoderReranker()
    
    print("\nIndexing documents...")
    print("  Building BM25 index...")
    bm25_retriever.index(documents)
    print("  Building dense index...")
    dense_retriever.index(documents)
    
    evaluator = RetrievalEvaluator()
    results: Dict[str, Any] = {
        "experiment_config": {
            "num_queries": len(queries),
            "num_documents": len(documents),
            "first_stage_k_values": first_stage_k_values
        },
        "results": {},
        "latency_analysis": {},
        "recommendations": {}
    }
    
    print("\n" + "-" * 60)
    print("Evaluating BM25 (no reranking)...")
    bm25_pipeline = TwoStageRetrievalPipeline(bm25_retriever, reranker=None, first_stage_k=100, final_k=10)
    bm25_metrics = evaluator.evaluate(bm25_pipeline, queries, qrels)
    results["results"]["bm25_only"] = bm25_metrics.to_dict()
    print(f"  MRR: {bm25_metrics.mrr:.4f}, NDCG@10: {bm25_metrics.ndcg_at_10:.4f}, "
          f"Recall@10: {bm25_metrics.recall_at_10:.4f}")
    print(f"  Avg retrieval time: {bm25_metrics.avg_retrieval_time_ms:.2f}ms")
    
    print("\n" + "-" * 60)
    print("Evaluating Dense Retrieval (no reranking)...")
    dense_pipeline = TwoStageRetrievalPipeline(dense_retriever, reranker=None, first_stage_k=100, final_k=10)
    dense_metrics = evaluator.evaluate(dense_pipeline, queries, qrels)
    results["results"]["dense_only"] = dense_metrics.to_dict()
    print(f"  MRR: {dense_metrics.mrr:.4f}, NDCG@10: {dense_metrics.ndcg_at_10:.4f}, "
          f"Recall@10: {dense_metrics.recall_at_10:.4f}")
    print(f"  Avg retrieval time: {dense_metrics.avg_retrieval_time_ms:.2f}ms")
    
    print("\n" + "-" * 60)
    print("Evaluating BM25 + Cross-Encoder Reranking...")
    bm25_rerank_results = {}
    for k in first_stage_k_values:
        print(f"\n  First-stage k={k}:")
        pipeline = TwoStageRetrievalPipeline(
            bm25_retriever, reranker=cross_encoder, first_stage_k=k, final_k=10
        )
        metrics = evaluator.evaluate(pipeline, queries, qrels)
        bm25_rerank_results[f"k={k}"] = metrics.to_dict()
        print(f"    MRR: {metrics.mrr:.4f}, NDCG@10: {metrics.ndcg_at_10:.4f}, "
              f"Recall@10: {metrics.recall_at_10:.4f}")
        print(f"    Avg retrieval: {metrics.avg_retrieval_time_ms:.2f}ms, "
              f"Avg reranking: {metrics.avg_reranking_time_ms:.2f}ms")
    results["results"]["bm25_with_reranking"] = bm25_rerank_results
    
    print("\n" + "-" * 60)
    print("Evaluating Dense + Cross-Encoder Reranking...")
    dense_rerank_results = {}
    for k in first_stage_k_values:
        print(f"\n  First-stage k={k}:")
        pipeline = TwoStageRetrievalPipeline(
            dense_retriever, reranker=cross_encoder, first_stage_k=k, final_k=10
        )
        metrics = evaluator.evaluate(pipeline, queries, qrels)
        dense_rerank_results[f"k={k}"] = metrics.to_dict()
        print(f"    MRR: {metrics.mrr:.4f}, NDCG@10: {metrics.ndcg_at_10:.4f}, "
              f"Recall@10: {metrics.recall_at_10:.4f}")
        print(f"    Avg retrieval: {metrics.avg_retrieval_time_ms:.2f}ms, "
              f"Avg reranking: {metrics.avg_reranking_time_ms:.2f}ms")
    results["results"]["dense_with_reranking"] = dense_rerank_results
    
    print("\n" + "=" * 60)
    print("LATENCY VS ACCURACY ANALYSIS")
    print("=" * 60)
    
    latency_analysis = analyze_latency_tradeoffs(results["results"])
    results["latency_analysis"] = latency_analysis
    
    recommendations = generate_recommendations(results["results"], latency_analysis)
    results["recommendations"] = recommendations
    
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


def analyze_latency_tradeoffs(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze latency vs accuracy tradeoffs from experiment results."""
    analysis = {
        "bm25_baseline": {
            "latency_ms": results["bm25_only"]["avg_retrieval_time_ms"],
            "mrr": results["bm25_only"]["mrr"],
            "ndcg@10": results["bm25_only"]["ndcg@10"]
        },
        "dense_baseline": {
            "latency_ms": results["dense_only"]["avg_retrieval_time_ms"],
            "mrr": results["dense_only"]["mrr"],
            "ndcg@10": results["dense_only"]["ndcg@10"]
        },
        "reranking_overhead": {},
        "accuracy_improvements": {}
    }
    
    for k_config, metrics in results.get("bm25_with_reranking", {}).items():
        total_latency = metrics["avg_retrieval_time_ms"] + metrics["avg_reranking_time_ms"]
        baseline_latency = results["bm25_only"]["avg_retrieval_time_ms"]
        overhead = total_latency - baseline_latency
        
        mrr_improvement = metrics["mrr"] - results["bm25_only"]["mrr"]
        ndcg_improvement = metrics["ndcg@10"] - results["bm25_only"]["ndcg@10"]
        
        analysis["reranking_overhead"][f"bm25_{k_config}"] = {
            "total_latency_ms": total_latency,
            "reranking_overhead_ms": overhead,
            "overhead_percentage": (overhead / baseline_latency * 100) if baseline_latency > 0 else 0
        }
        
        analysis["accuracy_improvements"][f"bm25_{k_config}"] = {
            "mrr_improvement": mrr_improvement,
            "ndcg_improvement": ndcg_improvement,
            "mrr_improvement_percentage": (mrr_improvement / results["bm25_only"]["mrr"] * 100) 
                                          if results["bm25_only"]["mrr"] > 0 else 0
        }
    
    for k_config, metrics in results.get("dense_with_reranking", {}).items():
        total_latency = metrics["avg_retrieval_time_ms"] + metrics["avg_reranking_time_ms"]
        baseline_latency = results["dense_only"]["avg_retrieval_time_ms"]
        overhead = total_latency - baseline_latency
        
        mrr_improvement = metrics["mrr"] - results["dense_only"]["mrr"]
        ndcg_improvement = metrics["ndcg@10"] - results["dense_only"]["ndcg@10"]
        
        analysis["reranking_overhead"][f"dense_{k_config}"] = {
            "total_latency_ms": total_latency,
            "reranking_overhead_ms": overhead,
            "overhead_percentage": (overhead / baseline_latency * 100) if baseline_latency > 0 else 0
        }
        
        analysis["accuracy_improvements"][f"dense_{k_config}"] = {
            "mrr_improvement": mrr_improvement,
            "ndcg_improvement": ndcg_improvement,
            "mrr_improvement_percentage": (mrr_improvement / results["dense_only"]["mrr"] * 100)
                                          if results["dense_only"]["mrr"] > 0 else 0
        }
    
    return analysis


def generate_recommendations(results: Dict[str, Any], latency_analysis: Dict[str, Any]) -> Dict[str, str]:
    """Generate recommendations based on experiment results."""
    recommendations = {}
    
    best_mrr = 0
    best_config = ""
    for config_name in ["bm25_only", "dense_only"]:
        if results[config_name]["mrr"] > best_mrr:
            best_mrr = results[config_name]["mrr"]
            best_config = config_name
    
    for retriever_type in ["bm25_with_reranking", "dense_with_reranking"]:
        for k_config, metrics in results.get(retriever_type, {}).items():
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_config = f"{retriever_type}_{k_config}"
    
    recommendations["best_accuracy_config"] = best_config
    recommendations["best_mrr"] = f"{best_mrr:.4f}"
    
    best_efficiency = None
    best_efficiency_config = ""
    for config, data in latency_analysis.get("accuracy_improvements", {}).items():
        overhead_data = latency_analysis["reranking_overhead"].get(config, {})
        if overhead_data.get("reranking_overhead_ms", 0) > 0:
            efficiency = data["mrr_improvement"] / overhead_data["reranking_overhead_ms"]
            if best_efficiency is None or efficiency > best_efficiency:
                best_efficiency = efficiency
                best_efficiency_config = config
    
    if best_efficiency_config:
        recommendations["best_efficiency_config"] = best_efficiency_config
        recommendations["efficiency_explanation"] = (
            f"Best MRR improvement per millisecond of latency overhead"
        )
    
    bm25_rerank = results.get("bm25_with_reranking", {})
    if bm25_rerank:
        k_values = sorted([int(k.split("=")[1]) for k in bm25_rerank.keys()])
        mrr_values = [bm25_rerank[f"k={k}"]["mrr"] for k in k_values]
        
        optimal_k = k_values[0]
        for i in range(1, len(k_values)):
            improvement = mrr_values[i] - mrr_values[i-1]
            if improvement < 0.005:
                optimal_k = k_values[i-1]
                break
            optimal_k = k_values[i]
        
        recommendations["optimal_candidates_to_rerank"] = str(optimal_k)
        recommendations["optimal_k_explanation"] = (
            f"Beyond k={optimal_k}, diminishing returns on accuracy improvement"
        )
    
    return recommendations


if __name__ == "__main__":
    results = run_experiments(
        num_queries=50,
        num_docs=5000,
        first_stage_k_values=[20, 50, 100, 200],
        output_file="reranking_results.json"
    )
