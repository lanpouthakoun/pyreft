"""
Hybrid Retriever

Combines BM25 and dense retrieval with configurable fusion strategies.
"""

from typing import List, Tuple, Optional, Union
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .fusion import BaseFusion, RRFFusion, WeightedFusion, LearnedFusion


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense vector search.

    Supports multiple fusion strategies for combining results.
    """

    def __init__(
        self,
        dense_model_name: str = "all-MiniLM-L6-v2",
        fusion_strategy: Union[str, BaseFusion] = "rrf",
        fusion_params: Optional[dict] = None,
        documents: Optional[List[str]] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            dense_model_name: Name of sentence-transformer model
            fusion_strategy: Fusion strategy ("rrf", "weighted", "learned") or BaseFusion instance
            fusion_params: Parameters for the fusion strategy
            documents: Optional list of documents to index
        """
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(model_name=dense_model_name)
        self.documents: List[str] = []

        fusion_params = fusion_params or {}
        if isinstance(fusion_strategy, BaseFusion):
            self.fusion = fusion_strategy
        elif fusion_strategy == "rrf":
            self.fusion = RRFFusion(**fusion_params)
        elif fusion_strategy == "weighted":
            self.fusion = WeightedFusion(**fusion_params)
        elif fusion_strategy == "learned":
            self.fusion = LearnedFusion(**fusion_params)
        else:
            raise ValueError(
                f"Unknown fusion strategy: {fusion_strategy}. "
                "Use 'rrf', 'weighted', 'learned', or a BaseFusion instance."
            )

        if documents:
            self.index(documents)

    def index(self, documents: List[str]) -> None:
        """
        Index documents for both BM25 and dense retrieval.

        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.bm25_retriever.index(documents)
        self.dense_retriever.index(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
    ) -> List[Tuple[int, float, str]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query
            top_k: Number of final results to return
            bm25_top_k: Number of BM25 candidates (default: 2*top_k)
            dense_top_k: Number of dense candidates (default: 2*top_k)

        Returns:
            List of tuples (doc_index, score, document_text)
        """
        if not self.documents:
            raise ValueError("No documents indexed. Call index() first.")

        bm25_top_k = bm25_top_k or (2 * top_k)
        dense_top_k = dense_top_k or (2 * top_k)

        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_top_k)

        return self.fusion.fuse(
            bm25_results, dense_results, self.documents, top_k=top_k
        )

    def retrieve_bm25_only(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Retrieve using only BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of tuples (doc_index, score, document_text)
        """
        return self.bm25_retriever.retrieve(query, top_k=top_k)

    def retrieve_dense_only(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Retrieve using only dense search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of tuples (doc_index, score, document_text)
        """
        return self.dense_retriever.retrieve(query, top_k=top_k)

    def get_all_scores(self, query: str) -> Tuple[List[float], List[float]]:
        """
        Get scores from both retrievers for all documents.

        Args:
            query: Search query

        Returns:
            Tuple of (bm25_scores, dense_scores)
        """
        bm25_scores = self.bm25_retriever.get_scores(query)
        dense_scores = self.dense_retriever.get_scores(query)
        return bm25_scores, dense_scores

    def set_fusion_strategy(
        self,
        fusion_strategy: Union[str, BaseFusion],
        fusion_params: Optional[dict] = None,
    ) -> None:
        """
        Change the fusion strategy.

        Args:
            fusion_strategy: New fusion strategy
            fusion_params: Parameters for the fusion strategy
        """
        fusion_params = fusion_params or {}
        if isinstance(fusion_strategy, BaseFusion):
            self.fusion = fusion_strategy
        elif fusion_strategy == "rrf":
            self.fusion = RRFFusion(**fusion_params)
        elif fusion_strategy == "weighted":
            self.fusion = WeightedFusion(**fusion_params)
        elif fusion_strategy == "learned":
            self.fusion = LearnedFusion(**fusion_params)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    def compare_methods(
        self, query: str, top_k: int = 10
    ) -> dict:
        """
        Compare results from different retrieval methods.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Dictionary with results from each method
        """
        return {
            "bm25": self.retrieve_bm25_only(query, top_k),
            "dense": self.retrieve_dense_only(query, top_k),
            "hybrid": self.retrieve(query, top_k),
        }
