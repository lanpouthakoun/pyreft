"""
HyDE (Hypothetical Document Embeddings) Retrieval Pipeline

This module implements the HyDE retrieval approach where:
1. A query is used to generate a hypothetical document using an LLM
2. The hypothetical document is embedded
3. The embedding is used to search against corpus embeddings

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

import json
import time
import hashlib
import os
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class HyDEConfig:
    """Configuration for HyDE pipeline."""
    
    llm_model_name: str = "google/flan-t5-base"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: Optional[str] = None
    max_new_tokens: int = 256
    num_hypothetical_docs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    use_cache: bool = True


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    
    query_id: str
    query: str
    retrieved_doc_ids: list[str]
    retrieved_scores: list[float]
    relevant_doc_ids: list[str]
    hypothetical_doc: Optional[str] = None
    generation_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0


class HypotheticalDocumentGenerator:
    """Generates hypothetical documents from queries using an LLM."""
    
    def __init__(self, config: HyDEConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        if config.use_cache and config.cache_dir:
            self.cache_path = Path(config.cache_dir) / "hyde_cache.json"
            self.cache = self._load_cache()
        else:
            self.cache_path = None
            self.cache = {}
    
    def _load_model(self):
        """Load the LLM for hypothetical document generation."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        print(f"Loading LLM: {self.config.llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.llm_model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
        ).to(self.config.device)
        self.model.eval()
    
    def _load_cache(self) -> dict:
        """Load cached hypothetical documents."""
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save hypothetical documents to cache."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _create_prompt(self, query: str) -> str:
        """Create a prompt for hypothetical document generation."""
        return (
            f"Write a detailed passage that would be a relevant answer to the following question. "
            f"The passage should contain factual information that directly addresses the question.\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )
    
    def generate(self, query: str) -> tuple[str, float]:
        """
        Generate a hypothetical document for a query.
        
        Returns:
            Tuple of (hypothetical_document, generation_time_ms)
        """
        cache_key = self._get_cache_key(query)
        
        if self.config.use_cache and cache_key in self.cache:
            return self.cache[cache_key], 0.0
        
        prompt = self._create_prompt(query)
        
        start_time = time.time()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        hypothetical_doc = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time_ms = (time.time() - start_time) * 1000
        
        if self.config.use_cache:
            self.cache[cache_key] = hypothetical_doc
            self._save_cache()
        
        return hypothetical_doc, generation_time_ms
    
    def generate_batch(self, queries: list[str]) -> tuple[list[str], list[float]]:
        """Generate hypothetical documents for a batch of queries."""
        hypothetical_docs = []
        generation_times = []
        
        for query in tqdm(queries, desc="Generating hypothetical documents"):
            doc, gen_time = self.generate(query)
            hypothetical_docs.append(doc)
            generation_times.append(gen_time)
        
        return hypothetical_docs, generation_times


class DocumentEmbedder:
    """Embeds documents using a sentence transformer model."""
    
    def __init__(self, config: HyDEConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {self.config.embedding_model_name}")
        self.model = SentenceTransformer(
            self.config.embedding_model_name,
            device=self.config.device
        )
    
    def embed(self, texts: list[str], show_progress: bool = True) -> tuple[np.ndarray, float]:
        """
        Embed a list of texts.
        
        Returns:
            Tuple of (embeddings, embedding_time_ms)
        """
        start_time = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embedding_time_ms = (time.time() - start_time) * 1000
        return embeddings, embedding_time_ms
    
    def embed_single(self, text: str) -> tuple[np.ndarray, float]:
        """Embed a single text."""
        return self.embed([text], show_progress=False)


class DenseRetriever:
    """Dense retrieval using cosine similarity."""
    
    def __init__(self, corpus_embeddings: np.ndarray, doc_ids: list[str]):
        self.corpus_embeddings = corpus_embeddings
        self.doc_ids = doc_ids
    
    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> tuple[list[str], list[float], float]:
        """
        Search for the most similar documents.
        
        Returns:
            Tuple of (doc_ids, scores, search_time_ms)
        """
        start_time = time.time()
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores = np.dot(query_embedding, self.corpus_embeddings.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved_doc_ids = [self.doc_ids[i] for i in top_indices]
        retrieved_scores = [float(scores[i]) for i in top_indices]
        
        search_time_ms = (time.time() - start_time) * 1000
        return retrieved_doc_ids, retrieved_scores, search_time_ms


class RetrievalMetrics:
    """Compute retrieval evaluation metrics."""
    
    @staticmethod
    def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
        """Compute reciprocal rank."""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
        """Compute recall at k."""
        if not relevant:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_at_k & relevant_set) / len(relevant_set)
    
    @staticmethod
    def dcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
        """Compute DCG at k."""
        relevant_set = set(relevant)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
        """Compute NDCG at k."""
        dcg = RetrievalMetrics.dcg_at_k(retrieved, relevant, k)
        ideal_retrieved = relevant[:k]
        idcg = RetrievalMetrics.dcg_at_k(ideal_retrieved, relevant, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def compute_all_metrics(results: list[RetrievalResult]) -> dict:
        """Compute all metrics across a list of results."""
        mrr_scores = []
        ndcg_10_scores = []
        recall_10_scores = []
        recall_100_scores = []
        
        for result in results:
            mrr_scores.append(
                RetrievalMetrics.reciprocal_rank(result.retrieved_doc_ids, result.relevant_doc_ids)
            )
            ndcg_10_scores.append(
                RetrievalMetrics.ndcg_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, 10)
            )
            recall_10_scores.append(
                RetrievalMetrics.recall_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, 10)
            )
            recall_100_scores.append(
                RetrievalMetrics.recall_at_k(result.retrieved_doc_ids, result.relevant_doc_ids, 100)
            )
        
        return {
            "MRR": float(np.mean(mrr_scores)),
            "NDCG@10": float(np.mean(ndcg_10_scores)),
            "Recall@10": float(np.mean(recall_10_scores)),
            "Recall@100": float(np.mean(recall_100_scores)),
            "num_queries": len(results)
        }


class HyDEPipeline:
    """Main HyDE retrieval pipeline."""
    
    def __init__(self, config: Optional[HyDEConfig] = None):
        self.config = config or HyDEConfig()
        self.generator = None
        self.embedder = None
        self.retriever = None
        self.corpus_embeddings = None
        self.doc_ids = None
    
    def initialize(self):
        """Initialize the pipeline components."""
        self.generator = HypotheticalDocumentGenerator(self.config)
        self.embedder = DocumentEmbedder(self.config)
    
    def index_corpus(self, documents: list[str], doc_ids: list[str]):
        """
        Index a corpus of documents.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
        """
        print(f"Indexing {len(documents)} documents...")
        self.doc_ids = doc_ids
        self.corpus_embeddings, _ = self.embedder.embed(documents, show_progress=True)
        self.retriever = DenseRetriever(self.corpus_embeddings, doc_ids)
        print(f"Indexed {len(documents)} documents with embedding shape {self.corpus_embeddings.shape}")
    
    def retrieve_hyde(
        self,
        query: str,
        query_id: str,
        relevant_doc_ids: list[str],
        top_k: int = 100
    ) -> RetrievalResult:
        """
        Retrieve documents using HyDE.
        
        Args:
            query: The query string
            query_id: Unique identifier for the query
            relevant_doc_ids: List of relevant document IDs (ground truth)
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with retrieved documents and timing information
        """
        hypothetical_doc, generation_time = self.generator.generate(query)
        query_embedding, embedding_time = self.embedder.embed_single(hypothetical_doc)
        retrieved_ids, retrieved_scores, search_time = self.retriever.search(
            query_embedding[0], top_k=top_k
        )
        
        return RetrievalResult(
            query_id=query_id,
            query=query,
            retrieved_doc_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_doc_ids=relevant_doc_ids,
            hypothetical_doc=hypothetical_doc,
            generation_time_ms=generation_time,
            embedding_time_ms=embedding_time,
            search_time_ms=search_time
        )
    
    def retrieve_standard(
        self,
        query: str,
        query_id: str,
        relevant_doc_ids: list[str],
        top_k: int = 100
    ) -> RetrievalResult:
        """
        Retrieve documents using standard dense retrieval (no HyDE).
        
        Args:
            query: The query string
            query_id: Unique identifier for the query
            relevant_doc_ids: List of relevant document IDs (ground truth)
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with retrieved documents and timing information
        """
        query_embedding, embedding_time = self.embedder.embed_single(query)
        retrieved_ids, retrieved_scores, search_time = self.retriever.search(
            query_embedding[0], top_k=top_k
        )
        
        return RetrievalResult(
            query_id=query_id,
            query=query,
            retrieved_doc_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_doc_ids=relevant_doc_ids,
            hypothetical_doc=None,
            generation_time_ms=0.0,
            embedding_time_ms=embedding_time,
            search_time_ms=search_time
        )
    
    def evaluate(
        self,
        queries: list[str],
        query_ids: list[str],
        relevant_docs: list[list[str]],
        top_k: int = 100
    ) -> dict:
        """
        Evaluate both HyDE and standard retrieval on a set of queries.
        
        Args:
            queries: List of query strings
            query_ids: List of query IDs
            relevant_docs: List of lists of relevant document IDs for each query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with evaluation results for both methods
        """
        hyde_results = []
        standard_results = []
        
        print("\nRunning HyDE retrieval...")
        for query, qid, rel_docs in tqdm(
            zip(queries, query_ids, relevant_docs),
            total=len(queries),
            desc="HyDE retrieval"
        ):
            result = self.retrieve_hyde(query, qid, rel_docs, top_k)
            hyde_results.append(result)
        
        print("\nRunning standard dense retrieval...")
        for query, qid, rel_docs in tqdm(
            zip(queries, query_ids, relevant_docs),
            total=len(queries),
            desc="Standard retrieval"
        ):
            result = self.retrieve_standard(query, qid, rel_docs, top_k)
            standard_results.append(result)
        
        hyde_metrics = RetrievalMetrics.compute_all_metrics(hyde_results)
        standard_metrics = RetrievalMetrics.compute_all_metrics(standard_results)
        
        hyde_latency = {
            "avg_generation_time_ms": float(np.mean([r.generation_time_ms for r in hyde_results])),
            "avg_embedding_time_ms": float(np.mean([r.embedding_time_ms for r in hyde_results])),
            "avg_search_time_ms": float(np.mean([r.search_time_ms for r in hyde_results])),
            "total_avg_latency_ms": float(np.mean([
                r.generation_time_ms + r.embedding_time_ms + r.search_time_ms
                for r in hyde_results
            ]))
        }
        
        standard_latency = {
            "avg_embedding_time_ms": float(np.mean([r.embedding_time_ms for r in standard_results])),
            "avg_search_time_ms": float(np.mean([r.search_time_ms for r in standard_results])),
            "total_avg_latency_ms": float(np.mean([
                r.embedding_time_ms + r.search_time_ms
                for r in standard_results
            ]))
        }
        
        per_query_analysis = self._analyze_per_query(hyde_results, standard_results)
        
        return {
            "hyde": {
                "metrics": hyde_metrics,
                "latency": hyde_latency,
                "per_query_results": [
                    {
                        "query_id": r.query_id,
                        "query": r.query,
                        "hypothetical_doc": r.hypothetical_doc,
                        "mrr": RetrievalMetrics.reciprocal_rank(r.retrieved_doc_ids, r.relevant_doc_ids),
                        "ndcg@10": RetrievalMetrics.ndcg_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, 10),
                        "recall@10": RetrievalMetrics.recall_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, 10),
                        "recall@100": RetrievalMetrics.recall_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, 100),
                        "generation_time_ms": r.generation_time_ms,
                        "top_10_retrieved": r.retrieved_doc_ids[:10]
                    }
                    for r in hyde_results
                ]
            },
            "standard": {
                "metrics": standard_metrics,
                "latency": standard_latency,
                "per_query_results": [
                    {
                        "query_id": r.query_id,
                        "query": r.query,
                        "mrr": RetrievalMetrics.reciprocal_rank(r.retrieved_doc_ids, r.relevant_doc_ids),
                        "ndcg@10": RetrievalMetrics.ndcg_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, 10),
                        "recall@10": RetrievalMetrics.recall_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, 10),
                        "recall@100": RetrievalMetrics.recall_at_k(r.retrieved_doc_ids, r.relevant_doc_ids, 100),
                        "top_10_retrieved": r.retrieved_doc_ids[:10]
                    }
                    for r in standard_results
                ]
            },
            "analysis": per_query_analysis
        }
    
    def _analyze_per_query(
        self,
        hyde_results: list[RetrievalResult],
        standard_results: list[RetrievalResult]
    ) -> dict:
        """Analyze when HyDE helps vs hurts performance."""
        hyde_wins = []
        standard_wins = []
        ties = []
        
        for hyde_r, std_r in zip(hyde_results, standard_results):
            hyde_mrr = RetrievalMetrics.reciprocal_rank(hyde_r.retrieved_doc_ids, hyde_r.relevant_doc_ids)
            std_mrr = RetrievalMetrics.reciprocal_rank(std_r.retrieved_doc_ids, std_r.relevant_doc_ids)
            
            query_info = {
                "query_id": hyde_r.query_id,
                "query": hyde_r.query,
                "hyde_mrr": hyde_mrr,
                "standard_mrr": std_mrr,
                "mrr_diff": hyde_mrr - std_mrr,
                "hypothetical_doc": hyde_r.hypothetical_doc
            }
            
            if hyde_mrr > std_mrr:
                hyde_wins.append(query_info)
            elif std_mrr > hyde_mrr:
                standard_wins.append(query_info)
            else:
                ties.append(query_info)
        
        hyde_wins.sort(key=lambda x: x["mrr_diff"], reverse=True)
        standard_wins.sort(key=lambda x: x["mrr_diff"])
        
        return {
            "summary": {
                "hyde_wins": len(hyde_wins),
                "standard_wins": len(standard_wins),
                "ties": len(ties),
                "total_queries": len(hyde_results),
                "hyde_win_rate": len(hyde_wins) / len(hyde_results) if hyde_results else 0,
                "avg_mrr_improvement_when_hyde_wins": float(np.mean([q["mrr_diff"] for q in hyde_wins])) if hyde_wins else 0,
                "avg_mrr_degradation_when_hyde_loses": float(np.mean([q["mrr_diff"] for q in standard_wins])) if standard_wins else 0
            },
            "hyde_wins_examples": hyde_wins[:5],
            "standard_wins_examples": standard_wins[:5],
            "analysis_notes": self._generate_analysis_notes(hyde_wins, standard_wins, ties)
        }
    
    def _generate_analysis_notes(
        self,
        hyde_wins: list[dict],
        standard_wins: list[dict],
        ties: list[dict]
    ) -> list[str]:
        """Generate analysis notes about when HyDE helps vs hurts."""
        notes = []
        
        total = len(hyde_wins) + len(standard_wins) + len(ties)
        if total == 0:
            return ["No queries to analyze"]
        
        hyde_win_rate = len(hyde_wins) / total
        
        if hyde_win_rate > 0.6:
            notes.append(
                f"HyDE significantly outperforms standard retrieval, winning on {hyde_win_rate:.1%} of queries. "
                "This suggests the hypothetical document generation is effectively bridging the vocabulary gap "
                "between queries and documents."
            )
        elif hyde_win_rate < 0.4:
            notes.append(
                f"Standard retrieval outperforms HyDE, with HyDE only winning on {hyde_win_rate:.1%} of queries. "
                "This may indicate that the LLM is generating hypothetical documents that diverge from the "
                "actual relevant documents in the corpus."
            )
        else:
            notes.append(
                f"HyDE and standard retrieval perform comparably, with HyDE winning on {hyde_win_rate:.1%} of queries. "
                "The benefit of HyDE appears to be query-dependent."
            )
        
        if hyde_wins:
            avg_query_len_hyde_wins = np.mean([len(q["query"].split()) for q in hyde_wins])
            notes.append(
                f"When HyDE wins, queries have an average length of {avg_query_len_hyde_wins:.1f} words."
            )
        
        if standard_wins:
            avg_query_len_std_wins = np.mean([len(q["query"].split()) for q in standard_wins])
            notes.append(
                f"When standard retrieval wins, queries have an average length of {avg_query_len_std_wins:.1f} words."
            )
        
        notes.append(
            "HyDE tends to help more with short, keyword-style queries where the LLM can expand "
            "the query into a more descriptive passage. It may hurt performance when queries are "
            "already well-formed or when the LLM hallucinates incorrect information."
        )
        
        return notes


def create_sample_dataset() -> tuple[list[str], list[str], list[str], list[str], list[list[str]]]:
    """
    Create a sample dataset for testing the HyDE pipeline.
    
    Returns:
        Tuple of (documents, doc_ids, queries, query_ids, relevant_docs)
    """
    documents = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists "
        "and intellectuals for its design, but it has become a global cultural icon of France.",
        
        "Machine learning is a subset of artificial intelligence that provides systems the ability to "
        "automatically learn and improve from experience without being explicitly programmed. "
        "Machine learning focuses on the development of computer programs that can access data and "
        "use it to learn for themselves.",
        
        "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, "
        "and other materials, built along the historical northern borders of China to protect against "
        "various nomadic groups. Several walls were built from as early as the 7th century BC.",
        
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes "
        "code readability with the use of significant indentation. Python is dynamically typed and "
        "garbage-collected. It supports multiple programming paradigms, including structured, "
        "object-oriented and functional programming.",
        
        "The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome that covers "
        "most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which "
        "5,500,000 km2 are covered by the rainforest.",
        
        "Neural networks are computing systems inspired by biological neural networks that constitute "
        "animal brains. An artificial neural network is based on a collection of connected units or "
        "nodes called artificial neurons, which loosely model the neurons in a biological brain.",
        
        "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. "
        "Considered an archetypal masterpiece of the Italian Renaissance, it has been described as "
        "the best known, the most visited, the most written about, and the most parodied work of art.",
        
        "Deep learning is part of a broader family of machine learning methods based on artificial "
        "neural networks with representation learning. Learning can be supervised, semi-supervised "
        "or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural "
        "networks, and convolutional neural networks have been applied to many fields.",
        
        "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy. Built of "
        "travertine limestone, tuff, and brick-faced concrete, it was the largest amphitheatre ever "
        "built at the time and held 50,000 to 80,000 spectators.",
        
        "Natural language processing is a subfield of linguistics, computer science, and artificial "
        "intelligence concerned with the interactions between computers and human language, in particular "
        "how to program computers to process and analyze large amounts of natural language data."
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    queries = [
        "What is the Eiffel Tower?",
        "How does machine learning work?",
        "Tell me about the Great Wall",
        "What is Python programming?",
        "Where is the Amazon rainforest?",
        "What are neural networks?",
        "Who painted the Mona Lisa?",
        "What is deep learning?",
        "Describe the Colosseum",
        "What is NLP?"
    ]
    
    query_ids = [f"q_{i}" for i in range(len(queries))]
    
    relevant_docs = [
        ["doc_0"],
        ["doc_1"],
        ["doc_2"],
        ["doc_3"],
        ["doc_4"],
        ["doc_5"],
        ["doc_6"],
        ["doc_7"],
        ["doc_8"],
        ["doc_9"]
    ]
    
    return documents, doc_ids, queries, query_ids, relevant_docs


def create_challenging_dataset() -> tuple[list[str], list[str], list[str], list[str], list[list[str]]]:
    """
    Create a more challenging dataset that better demonstrates HyDE's strengths and weaknesses.
    
    This dataset includes:
    - More documents with overlapping topics
    - Keyword-style queries (where HyDE should help)
    - Well-formed queries (where standard retrieval may be sufficient)
    - Ambiguous queries (to test robustness)
    
    Returns:
        Tuple of (documents, doc_ids, queries, query_ids, relevant_docs)
    """
    documents = [
        "Photosynthesis is the process by which plants convert light energy into chemical energy. "
        "Chlorophyll in plant cells absorbs sunlight and uses it to transform carbon dioxide and water "
        "into glucose and oxygen. This process is essential for life on Earth as it produces oxygen "
        "and forms the base of most food chains.",
        
        "The mitochondria are often called the powerhouse of the cell because they generate most of "
        "the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. "
        "Mitochondria have their own DNA and are thought to have originated from ancient bacteria.",
        
        "Climate change refers to long-term shifts in global temperatures and weather patterns. "
        "Human activities, particularly burning fossil fuels, have been the main driver of climate "
        "change since the 1800s. Effects include rising sea levels, more frequent extreme weather, "
        "and disruptions to ecosystems.",
        
        "The greenhouse effect is a natural process that warms the Earth's surface. When the Sun's "
        "energy reaches the Earth, some is reflected back to space and some is absorbed. Greenhouse "
        "gases trap heat in the atmosphere, making Earth habitable.",
        
        "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement "
        "to process information. Unlike classical computers that use bits (0 or 1), quantum computers "
        "use qubits that can exist in multiple states simultaneously.",
        
        "Classical computing relies on transistors and binary logic gates to perform calculations. "
        "Modern processors contain billions of transistors and can execute billions of operations "
        "per second. Moore's Law predicted the doubling of transistors every two years.",
        
        "The Renaissance was a cultural movement that began in Italy in the 14th century and spread "
        "throughout Europe. It marked a renewed interest in classical Greek and Roman culture, "
        "leading to advances in art, architecture, literature, and science.",
        
        "The Industrial Revolution began in Britain in the late 18th century and transformed "
        "manufacturing processes. Steam power and mechanization replaced hand production methods, "
        "leading to urbanization and significant social changes.",
        
        "Antibiotics are medications used to treat bacterial infections. Alexander Fleming discovered "
        "penicillin in 1928, revolutionizing medicine. However, antibiotic resistance has become "
        "a major global health concern due to overuse and misuse.",
        
        "Vaccines work by training the immune system to recognize and fight specific pathogens. "
        "They contain weakened or inactive parts of a pathogen, triggering an immune response "
        "without causing the disease. Vaccination has eradicated smallpox and nearly eliminated polio.",
        
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, "
        "can escape. They form when massive stars collapse at the end of their life cycle. "
        "The event horizon marks the boundary beyond which escape is impossible.",
        
        "Dark matter is a hypothetical form of matter that doesn't emit or absorb light. "
        "It is thought to account for approximately 85% of the matter in the universe. "
        "Its existence is inferred from gravitational effects on visible matter.",
        
        "CRISPR-Cas9 is a revolutionary gene-editing technology that allows scientists to modify DNA "
        "sequences with unprecedented precision. It has applications in treating genetic diseases, "
        "developing new crops, and understanding gene function.",
        
        "Stem cells are undifferentiated cells capable of developing into many different cell types. "
        "They play a crucial role in development and tissue repair. Embryonic stem cells can become "
        "any cell type, while adult stem cells are more limited.",
        
        "The human microbiome consists of trillions of microorganisms living in and on our bodies. "
        "These bacteria, viruses, and fungi play essential roles in digestion, immunity, and even "
        "mental health. Disruptions to the microbiome are linked to various diseases.",
        
        "Blockchain is a distributed ledger technology that records transactions across multiple "
        "computers. Each block contains a cryptographic hash of the previous block, creating "
        "an immutable chain. Bitcoin was the first application of blockchain technology.",
        
        "Artificial neural networks are computing systems inspired by biological neural networks. "
        "They consist of interconnected nodes that process information using connectionist approaches. "
        "Deep learning uses multiple layers of neural networks to learn complex patterns.",
        
        "The theory of relativity, developed by Albert Einstein, describes the relationship between "
        "space, time, and gravity. Special relativity deals with objects moving at constant speeds, "
        "while general relativity explains gravity as the curvature of spacetime.",
        
        "Plate tectonics is the scientific theory explaining the movement of Earth's lithosphere. "
        "The Earth's surface is divided into plates that float on the semi-fluid asthenosphere. "
        "Plate movements cause earthquakes, volcanic activity, and mountain formation.",
        
        "The water cycle describes the continuous movement of water on, above, and below Earth's "
        "surface. It includes evaporation, condensation, precipitation, and collection. "
        "This cycle is essential for distributing fresh water around the planet."
    ]
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    queries = [
        "plants energy sun",
        "cell energy production",
        "global warming causes",
        "how does Earth stay warm",
        "quantum vs classical computing",
        "new type of computer",
        "European cultural rebirth",
        "factory machines history",
        "bacteria medicine",
        "disease prevention injection",
        "space gravity escape",
        "invisible universe matter",
        "DNA editing tool",
        "cells that can become anything",
        "gut bacteria health",
        "cryptocurrency technology",
        "brain-inspired computing",
        "Einstein space time",
        "Earth surface movement",
        "rain evaporation cycle"
    ]
    
    query_ids = [f"q_{i}" for i in range(len(queries))]
    
    relevant_docs = [
        ["doc_0"],
        ["doc_1"],
        ["doc_2"],
        ["doc_3"],
        ["doc_4", "doc_5"],
        ["doc_4"],
        ["doc_6"],
        ["doc_7"],
        ["doc_8"],
        ["doc_9"],
        ["doc_10"],
        ["doc_11"],
        ["doc_12"],
        ["doc_13"],
        ["doc_14"],
        ["doc_15"],
        ["doc_16"],
        ["doc_17"],
        ["doc_18"],
        ["doc_19"]
    ]
    
    return documents, doc_ids, queries, query_ids, relevant_docs


def run_evaluation(output_path: str = "hyde_results.json", use_challenging_dataset: bool = True):
    """Run the full HyDE evaluation pipeline."""
    print("=" * 60)
    print("HyDE Retrieval Pipeline Evaluation")
    print("=" * 60)
    
    config = HyDEConfig(
        llm_model_name="google/flan-t5-base",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="/tmp/hyde_cache",
        use_cache=True
    )
    
    print("\nInitializing pipeline...")
    pipeline = HyDEPipeline(config)
    pipeline.initialize()
    
    if use_challenging_dataset:
        print("\nCreating challenging dataset...")
        documents, doc_ids, queries, query_ids, relevant_docs = create_challenging_dataset()
    else:
        print("\nCreating sample dataset...")
        documents, doc_ids, queries, query_ids, relevant_docs = create_sample_dataset()
    
    print("\nIndexing corpus...")
    pipeline.index_corpus(documents, doc_ids)
    
    print("\nRunning evaluation...")
    results = pipeline.evaluate(queries, query_ids, relevant_docs)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nHyDE Retrieval Metrics:")
    for metric, value in results["hyde"]["metrics"].items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    print("\nStandard Dense Retrieval Metrics:")
    for metric, value in results["standard"]["metrics"].items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    print("\nLatency Comparison:")
    print(f"  HyDE total avg latency: {results['hyde']['latency']['total_avg_latency_ms']:.2f} ms")
    print(f"    - Generation: {results['hyde']['latency']['avg_generation_time_ms']:.2f} ms")
    print(f"    - Embedding: {results['hyde']['latency']['avg_embedding_time_ms']:.2f} ms")
    print(f"    - Search: {results['hyde']['latency']['avg_search_time_ms']:.2f} ms")
    print(f"  Standard total avg latency: {results['standard']['latency']['total_avg_latency_ms']:.2f} ms")
    
    print("\nAnalysis Summary:")
    analysis = results["analysis"]["summary"]
    print(f"  HyDE wins: {analysis['hyde_wins']} queries")
    print(f"  Standard wins: {analysis['standard_wins']} queries")
    print(f"  Ties: {analysis['ties']} queries")
    print(f"  HyDE win rate: {analysis['hyde_win_rate']:.1%}")
    
    print("\nAnalysis Notes:")
    for note in results["analysis"]["analysis_notes"]:
        print(f"  - {note}")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_evaluation()
