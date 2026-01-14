"""
Dense Retrieval Pipeline using Sentence Transformers and FAISS

This module implements a dense vector retrieval pipeline for document search
using sentence-transformers for embedding generation and FAISS for efficient
similarity search.

Evaluation metrics include:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Recall@K
"""

import json
import time
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class SyntheticDataset:
    """A synthetic dataset for retrieval evaluation."""
    documents: List[str]
    queries: List[str]
    relevance_labels: List[Dict[int, int]]

    def get_relevant_docs(self, query_idx: int) -> Set[int]:
        """Get relevant document indices for a query."""
        if query_idx >= len(self.relevance_labels):
            return set()
        return set(self.relevance_labels[query_idx].keys())


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    query_idx: int
    query: str
    retrieved_indices: List[int]
    retrieved_scores: List[float]
    relevant_indices: Set[int]
    mrr: float
    ndcg_10: float
    recall_10: float
    recall_100: float


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    model_name: str
    num_documents: int
    num_queries: int
    embedding_dim: int
    embedding_time_seconds: float
    avg_search_latency_ms: float
    mrr: float
    ndcg_10: float
    recall_10: float
    recall_100: float
    per_query_results: List[Dict] = field(default_factory=list)


class SyntheticDataGenerator:
    """Generator for synthetic documents and queries for retrieval evaluation."""

    TOPICS = {
        "machine_learning": {
            "keywords": [
                "neural network", "deep learning", "gradient descent", "backpropagation",
                "convolutional neural network", "recurrent neural network", "transformer",
                "attention mechanism", "embedding", "feature extraction", "classification",
                "regression", "clustering", "dimensionality reduction"
            ],
            "templates": [
                "{keyword} is a fundamental concept in modern {application}.",
                "The {keyword} technique has revolutionized {application}.",
                "Researchers use {keyword} to improve {application} performance.",
                "{keyword} combined with {keyword2} enables advanced {application}.",
                "Understanding {keyword} is essential for {application} practitioners."
            ],
            "applications": [
                "computer vision", "natural language processing", "speech recognition",
                "recommendation systems", "autonomous vehicles", "medical diagnosis",
                "fraud detection", "predictive analytics"
            ]
        },
        "databases": {
            "keywords": [
                "SQL", "NoSQL", "indexing", "query optimization", "transaction",
                "ACID properties", "sharding", "replication", "normalization",
                "denormalization", "B-tree", "hash index", "full-text search"
            ],
            "templates": [
                "{keyword} is critical for efficient {application}.",
                "Modern databases implement {keyword} for {application}.",
                "{keyword} ensures data integrity in {application}.",
                "The combination of {keyword} and {keyword2} optimizes {application}.",
                "Database administrators rely on {keyword} for {application}."
            ],
            "applications": [
                "data warehousing", "real-time analytics", "e-commerce platforms",
                "financial systems", "content management", "inventory management",
                "customer relationship management", "log analysis"
            ]
        },
        "web_development": {
            "keywords": [
                "REST API", "GraphQL", "microservices", "containerization",
                "load balancing", "caching", "CDN", "authentication",
                "authorization", "session management", "WebSocket", "HTTP/2"
            ],
            "templates": [
                "{keyword} enables scalable {application}.",
                "Web developers use {keyword} to build {application}.",
                "{keyword} improves the performance of {application}.",
                "Implementing {keyword} with {keyword2} enhances {application}.",
                "{keyword} is a best practice for modern {application}."
            ],
            "applications": [
                "single-page applications", "progressive web apps", "e-commerce sites",
                "social media platforms", "streaming services", "collaboration tools",
                "content delivery", "real-time messaging"
            ]
        },
        "cybersecurity": {
            "keywords": [
                "encryption", "firewall", "intrusion detection", "vulnerability scanning",
                "penetration testing", "zero trust", "multi-factor authentication",
                "security audit", "threat modeling", "incident response",
                "malware analysis", "network security", "endpoint protection"
            ],
            "templates": [
                "{keyword} is critical for protecting {application}.",
                "Organizations implement {keyword} to secure {application}.",
                "Advanced {keyword} techniques defend against {application} threats.",
                "{keyword} compliance is required for {application} systems.",
                "The integration of {keyword} and {keyword2} strengthens {application}."
            ],
            "applications": [
                "enterprise networks", "cloud infrastructure", "financial transactions",
                "healthcare data", "government systems", "critical infrastructure",
                "personal devices", "IoT networks", "supply chain systems"
            ]
        },
        "data_science": {
            "keywords": [
                "data visualization", "statistical analysis", "hypothesis testing",
                "A/B testing", "time series analysis", "anomaly detection",
                "feature engineering", "data cleaning", "ETL pipelines",
                "data warehousing", "business intelligence", "predictive modeling"
            ],
            "templates": [
                "{keyword} enables insights from {application} data.",
                "Data scientists use {keyword} for {application} analysis.",
                "{keyword} techniques reveal patterns in {application}.",
                "Effective {keyword} drives {application} decision making.",
                "Combining {keyword} with {keyword2} enhances {application} outcomes."
            ],
            "applications": [
                "customer behavior", "market trends", "operational efficiency",
                "risk assessment", "product development", "resource allocation",
                "performance metrics", "user engagement", "revenue optimization"
            ]
        },
        "cloud_computing": {
            "keywords": [
                "AWS", "Azure", "Google Cloud", "infrastructure as code",
                "auto-scaling", "serverless functions", "object storage",
                "virtual machines", "container orchestration", "service mesh",
                "cloud migration", "hybrid cloud", "multi-cloud strategy"
            ],
            "templates": [
                "{keyword} provides scalable infrastructure for {application}.",
                "Enterprises leverage {keyword} for {application} workloads.",
                "{keyword} reduces costs for {application} deployments.",
                "The shift to {keyword} transforms {application} operations.",
                "{keyword} integrated with {keyword2} optimizes {application}."
            ],
            "applications": [
                "enterprise applications", "development environments", "disaster recovery",
                "big data processing", "machine learning training", "web hosting",
                "database management", "file storage", "collaboration platforms"
            ]
        }
    }

    QUERY_TEMPLATES = {
        "what_is": [
            "What is {keyword}?",
            "Explain {keyword} in simple terms.",
            "How does {keyword} work?",
            "What are the basics of {keyword}?"
        ],
        "how_to": [
            "How to implement {keyword}?",
            "Best practices for {keyword}",
            "Guide to using {keyword}",
            "Steps to set up {keyword}"
        ],
        "comparison": [
            "{keyword} vs {keyword2}",
            "Difference between {keyword} and {keyword2}",
            "When to use {keyword} over {keyword2}",
            "Comparing {keyword} and {keyword2}"
        ],
        "application": [
            "Using {keyword} for {application}",
            "{keyword} in {application}",
            "How {keyword} improves {application}",
            "Benefits of {keyword} for {application}"
        ],
        "advanced": [
            "Advanced {keyword} techniques",
            "Optimizing {keyword} performance",
            "Scaling {keyword} systems",
            "Troubleshooting {keyword} issues"
        ]
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _generate_document(self, topic: str, num_sentences: int = 5) -> str:
        """Generate a single document on a given topic."""
        topic_data = self.TOPICS[topic]
        keywords = topic_data["keywords"]
        templates = topic_data["templates"]
        applications = topic_data["applications"]

        sentences = []
        for _ in range(num_sentences):
            template = random.choice(templates)
            keyword = random.choice(keywords)
            keyword2 = random.choice([k for k in keywords if k != keyword])
            application = random.choice(applications)

            sentence = template.format(
                keyword=keyword,
                keyword2=keyword2,
                application=application
            )
            sentences.append(sentence)

        return " ".join(sentences)

    def _generate_query(
        self, topic: str, doc_keywords: List[str]
    ) -> Tuple[str, str]:
        """Generate a query related to a topic and document keywords."""
        topic_data = self.TOPICS[topic]
        query_type = random.choice(list(self.QUERY_TEMPLATES.keys()))
        template = random.choice(self.QUERY_TEMPLATES[query_type])

        keyword = random.choice(doc_keywords) if doc_keywords else random.choice(
            topic_data["keywords"]
        )
        keyword2 = random.choice(
            [k for k in topic_data["keywords"] if k != keyword]
        )
        application = random.choice(topic_data["applications"])

        query = template.format(
            keyword=keyword,
            keyword2=keyword2,
            application=application
        )

        return query, query_type

    def _extract_keywords(self, document: str, topic: str) -> List[str]:
        """Extract topic keywords present in a document."""
        topic_keywords = self.TOPICS[topic]["keywords"]
        doc_lower = document.lower()
        return [kw for kw in topic_keywords if kw.lower() in doc_lower]

    def generate(
        self,
        num_documents: int = 200,
        num_queries: int = 100,
        docs_per_topic: Optional[int] = None,
        sentences_per_doc: int = 5,
        relevant_docs_per_query: int = 3
    ) -> SyntheticDataset:
        """
        Generate a synthetic dataset for retrieval evaluation.

        Args:
            num_documents: Total number of documents to generate
            num_queries: Number of queries to generate
            docs_per_topic: Documents per topic (auto-calculated if None)
            sentences_per_doc: Sentences per document
            relevant_docs_per_query: Number of relevant docs per query

        Returns:
            SyntheticDataset with documents, queries, and relevance labels
        """
        topics = list(self.TOPICS.keys())
        docs_per_topic = docs_per_topic or (num_documents // len(topics))

        documents: List[str] = []
        doc_topics: List[str] = []
        doc_keywords: List[List[str]] = []

        for topic in topics:
            for _ in range(docs_per_topic):
                doc = self._generate_document(topic, sentences_per_doc)
                documents.append(doc)
                doc_topics.append(topic)
                doc_keywords.append(self._extract_keywords(doc, topic))

        while len(documents) < num_documents:
            topic = random.choice(topics)
            doc = self._generate_document(topic, sentences_per_doc)
            documents.append(doc)
            doc_topics.append(topic)
            doc_keywords.append(self._extract_keywords(doc, topic))

        queries: List[str] = []
        relevance_labels: List[Dict[int, int]] = []

        for _ in range(num_queries):
            topic = random.choice(topics)

            topic_doc_indices = [
                i for i, t in enumerate(doc_topics) if t == topic
            ]

            if len(topic_doc_indices) < relevant_docs_per_query:
                relevant_indices = topic_doc_indices
            else:
                relevant_indices = random.sample(
                    topic_doc_indices, relevant_docs_per_query
                )

            if relevant_indices:
                primary_doc_idx = relevant_indices[0]
                query, _ = self._generate_query(
                    topic, doc_keywords[primary_doc_idx]
                )
            else:
                query, _ = self._generate_query(topic, [])

            relevance = {}
            for rank, doc_idx in enumerate(relevant_indices):
                relevance[doc_idx] = max(1, relevant_docs_per_query - rank)

            partially_relevant = [
                i for i in topic_doc_indices if i not in relevant_indices
            ]
            for doc_idx in random.sample(
                partially_relevant, min(2, len(partially_relevant))
            ):
                relevance[doc_idx] = 1

            queries.append(query)
            relevance_labels.append(relevance)

        return SyntheticDataset(
            documents=documents,
            queries=queries,
            relevance_labels=relevance_labels
        )


class RetrievalMetrics:
    """Calculator for retrieval evaluation metrics."""

    @staticmethod
    def reciprocal_rank(
        retrieved: List[int],
        relevant: Set[int]
    ) -> float:
        """
        Calculate Reciprocal Rank.

        RR = 1 / (rank of first relevant document)
        """
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def recall_at_k(
        retrieved: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# of relevant docs in top K) / (total # of relevant docs)
        """
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
        return relevant_in_top_k / len(relevant)

    @staticmethod
    def dcg_at_k(
        retrieved: List[int],
        relevance_scores: Dict[int, int],
        k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K
        """
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevance_scores.get(doc, 0)
            dcg += rel / np.log2(i + 2)
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved: List[int],
        relevance_scores: Dict[int, int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K
        """
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevance_scores, k)

        ideal_ranking = sorted(
            relevance_scores.values(), reverse=True
        )[:k]
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_ranking)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg


class DenseRetriever:
    """Dense vector retriever using sentence-transformers and FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True
    ):
        """
        Initialize the dense retriever.

        Args:
            model_name: Name of the sentence-transformer model to use
            use_faiss: Whether to use FAISS for efficient similarity search
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.embedding_dim: int = 0

    def index_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> float:
        """
        Index documents by computing their embeddings.

        Args:
            documents: List of documents to index
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Time taken to generate embeddings in seconds
        """
        self.documents = documents

        start_time = time.time()
        self.embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )
        embedding_time = time.time() - start_time

        self.embeddings = self.embeddings.astype(np.float32)
        self.embedding_dim = self.embeddings.shape[1]

        faiss.normalize_L2(self.embeddings)

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)

        return embedding_time

    def search(
        self,
        query: str,
        top_k: int = 100
    ) -> Tuple[List[int], List[float], float]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (document indices, scores, search latency in ms)
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        start_time = time.time()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            indices = indices[0].tolist()
            scores = scores[0].tolist()
        else:
            scores = np.dot(self.embeddings, query_embedding.T).flatten()
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            indices = sorted_indices.tolist()
            scores = scores[sorted_indices].tolist()

        latency_ms = (time.time() - start_time) * 1000

        return indices, scores, latency_ms


class DenseRetrievalPipeline:
    """Complete dense retrieval pipeline with evaluation."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
        seed: Optional[int] = 42
    ):
        """
        Initialize the pipeline.

        Args:
            model_name: Sentence transformer model name
            use_faiss: Whether to use FAISS
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.seed = seed
        self.retriever: Optional[DenseRetriever] = None
        self.dataset: Optional[SyntheticDataset] = None

    def generate_dataset(
        self,
        num_documents: int = 200,
        num_queries: int = 100
    ) -> SyntheticDataset:
        """Generate synthetic dataset for evaluation."""
        generator = SyntheticDataGenerator(seed=self.seed)
        self.dataset = generator.generate(
            num_documents=num_documents,
            num_queries=num_queries
        )
        return self.dataset

    def build_index(
        self,
        documents: Optional[List[str]] = None
    ) -> float:
        """
        Build the retrieval index.

        Args:
            documents: Documents to index (uses dataset if None)

        Returns:
            Embedding generation time in seconds
        """
        if documents is None:
            if self.dataset is None:
                raise ValueError("No dataset available. Call generate_dataset() first.")
            documents = self.dataset.documents

        self.retriever = DenseRetriever(
            model_name=self.model_name,
            use_faiss=self.use_faiss
        )
        return self.retriever.index_documents(documents)

    def evaluate(
        self,
        queries: Optional[List[str]] = None,
        relevance_labels: Optional[List[Dict[int, int]]] = None,
        top_k: int = 100
    ) -> EvaluationResults:
        """
        Evaluate retrieval performance.

        Args:
            queries: Queries to evaluate (uses dataset if None)
            relevance_labels: Relevance labels (uses dataset if None)
            top_k: Number of documents to retrieve per query

        Returns:
            EvaluationResults with metrics and per-query results
        """
        if self.retriever is None:
            raise ValueError("No index built. Call build_index() first.")

        if queries is None:
            if self.dataset is None:
                raise ValueError("No dataset available.")
            queries = self.dataset.queries
            relevance_labels = self.dataset.relevance_labels

        if relevance_labels is None:
            raise ValueError("relevance_labels required if queries provided.")

        per_query_results: List[RetrievalResult] = []
        total_latency = 0.0

        for query_idx, (query, relevance) in enumerate(zip(queries, relevance_labels)):
            indices, scores, latency = self.retriever.search(query, top_k=top_k)
            total_latency += latency

            relevant = set(relevance.keys())

            mrr = RetrievalMetrics.reciprocal_rank(indices, relevant)
            ndcg_10 = RetrievalMetrics.ndcg_at_k(indices, relevance, k=10)
            recall_10 = RetrievalMetrics.recall_at_k(indices, relevant, k=10)
            recall_100 = RetrievalMetrics.recall_at_k(indices, relevant, k=100)

            result = RetrievalResult(
                query_idx=query_idx,
                query=query,
                retrieved_indices=indices[:10],
                retrieved_scores=scores[:10],
                relevant_indices=relevant,
                mrr=mrr,
                ndcg_10=ndcg_10,
                recall_10=recall_10,
                recall_100=recall_100
            )
            per_query_results.append(result)

        avg_mrr = np.mean([r.mrr for r in per_query_results])
        avg_ndcg_10 = np.mean([r.ndcg_10 for r in per_query_results])
        avg_recall_10 = np.mean([r.recall_10 for r in per_query_results])
        avg_recall_100 = np.mean([r.recall_100 for r in per_query_results])
        avg_latency = total_latency / len(queries)

        per_query_dicts = []
        for r in per_query_results:
            per_query_dicts.append({
                "query_idx": r.query_idx,
                "query": r.query,
                "retrieved_indices": r.retrieved_indices,
                "retrieved_scores": r.retrieved_scores,
                "relevant_indices": list(r.relevant_indices),
                "mrr": r.mrr,
                "ndcg_10": r.ndcg_10,
                "recall_10": r.recall_10,
                "recall_100": r.recall_100
            })

        return EvaluationResults(
            model_name=self.model_name,
            num_documents=len(self.retriever.documents),
            num_queries=len(queries),
            embedding_dim=self.retriever.embedding_dim,
            embedding_time_seconds=0.0,
            avg_search_latency_ms=avg_latency,
            mrr=float(avg_mrr),
            ndcg_10=float(avg_ndcg_10),
            recall_10=float(avg_recall_10),
            recall_100=float(avg_recall_100),
            per_query_results=per_query_dicts
        )

    def run_full_evaluation(
        self,
        num_documents: int = 200,
        num_queries: int = 100,
        output_file: str = "dense_results.json"
    ) -> EvaluationResults:
        """
        Run complete evaluation pipeline.

        Args:
            num_documents: Number of documents to generate
            num_queries: Number of queries to generate
            output_file: Path to save results

        Returns:
            EvaluationResults
        """
        print(f"Generating synthetic dataset with {num_documents} documents and {num_queries} queries...")
        self.generate_dataset(num_documents=num_documents, num_queries=num_queries)

        print(f"Building index with model: {self.model_name}...")
        embedding_time = self.build_index()
        print(f"Embedding generation time: {embedding_time:.2f} seconds")

        print("Evaluating retrieval performance...")
        results = self.evaluate()
        results.embedding_time_seconds = embedding_time

        results_dict = {
            "model_name": results.model_name,
            "num_documents": results.num_documents,
            "num_queries": results.num_queries,
            "embedding_dim": results.embedding_dim,
            "embedding_time_seconds": results.embedding_time_seconds,
            "avg_search_latency_ms": results.avg_search_latency_ms,
            "metrics": {
                "mrr": results.mrr,
                "ndcg_10": results.ndcg_10,
                "recall_10": results.recall_10,
                "recall_100": results.recall_100
            },
            "per_query_results": results.per_query_results
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to {output_file}")
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Model: {results.model_name}")
        print(f"Documents: {results.num_documents}")
        print(f"Queries: {results.num_queries}")
        print(f"Embedding Dimension: {results.embedding_dim}")
        print(f"Embedding Time: {results.embedding_time_seconds:.2f}s")
        print(f"Avg Search Latency: {results.avg_search_latency_ms:.2f}ms")
        print("-" * 50)
        print(f"MRR: {results.mrr:.4f}")
        print(f"NDCG@10: {results.ndcg_10:.4f}")
        print(f"Recall@10: {results.recall_10:.4f}")
        print(f"Recall@100: {results.recall_100:.4f}")
        print("=" * 50)

        return results


def compare_models(
    model_names: List[str],
    num_documents: int = 200,
    num_queries: int = 100,
    output_file: str = "model_comparison.json"
) -> Dict[str, EvaluationResults]:
    """
    Compare multiple embedding models.

    Args:
        model_names: List of sentence-transformer model names
        num_documents: Number of documents
        num_queries: Number of queries
        output_file: Path to save comparison results

    Returns:
        Dictionary mapping model names to results
    """
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate(
        num_documents=num_documents,
        num_queries=num_queries
    )

    results = {}
    comparison_data = []

    for model_name in model_names:
        print(f"\n{'=' * 50}")
        print(f"Evaluating model: {model_name}")
        print("=" * 50)

        pipeline = DenseRetrievalPipeline(model_name=model_name, seed=42)
        pipeline.dataset = dataset

        embedding_time = pipeline.build_index()
        eval_results = pipeline.evaluate()
        eval_results.embedding_time_seconds = embedding_time

        results[model_name] = eval_results

        comparison_data.append({
            "model_name": model_name,
            "embedding_dim": eval_results.embedding_dim,
            "embedding_time_seconds": eval_results.embedding_time_seconds,
            "avg_search_latency_ms": eval_results.avg_search_latency_ms,
            "mrr": eval_results.mrr,
            "ndcg_10": eval_results.ndcg_10,
            "recall_10": eval_results.recall_10,
            "recall_100": eval_results.recall_100
        })

    with open(output_file, "w") as f:
        json.dump({
            "num_documents": num_documents,
            "num_queries": num_queries,
            "models": comparison_data
        }, f, indent=2)

    print(f"\nComparison results saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dense Retrieval Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=200,
        help="Number of documents to generate"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dense_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple embedding models"
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Disable FAISS (use numpy for similarity search)"
    )

    args = parser.parse_args()

    if args.compare_models:
        models_to_compare = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2"
        ]
        compare_models(
            model_names=models_to_compare,
            num_documents=args.num_documents,
            num_queries=args.num_queries,
            output_file="model_comparison.json"
        )
    else:
        pipeline = DenseRetrievalPipeline(
            model_name=args.model,
            use_faiss=not args.no_faiss,
            seed=42
        )
        pipeline.run_full_evaluation(
            num_documents=args.num_documents,
            num_queries=args.num_queries,
            output_file=args.output
        )
