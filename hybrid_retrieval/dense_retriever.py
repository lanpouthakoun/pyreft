"""
Dense Retriever Component

Implements dense vector search using sentence-transformers embeddings.
"""

from typing import List, Tuple, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DenseRetriever:
    """Dense vector retriever using sentence-transformers embeddings."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        documents: Optional[List[str]] = None,
    ):
        """
        Initialize the dense retriever.

        Args:
            model_name: Name of the sentence-transformer model to use
            documents: Optional list of documents to index
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for DenseRetriever. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        if documents:
            self.index(documents)

    def index(self, documents: List[str]) -> None:
        """
        Index documents by computing their embeddings.

        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.embeddings = self.model.encode(
            documents, convert_to_numpy=True, show_progress_bar=False
        )
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def _compute_similarity(
        self, query_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and all documents.

        Args:
            query_embedding: Normalized query embedding

        Returns:
            Array of similarity scores
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index() first.")

        return np.dot(self.embeddings, query_embedding)

    def retrieve(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of tuples (doc_index, score, document_text)
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index() first.")

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = self._compute_similarity(query_embedding)

        scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scored_docs[:top_k]:
            results.append((doc_idx, score, self.documents[doc_idx]))

        return results

    def get_scores(self, query: str) -> List[float]:
        """
        Get similarity scores for all documents.

        Args:
            query: Search query

        Returns:
            List of scores for each document
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index() first.")

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = self._compute_similarity(query_embedding)
        return [float(s) for s in scores]

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector.

        Args:
            query: Query text

        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )[0]
        return embedding / np.linalg.norm(embedding)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents into embedding vectors.

        Args:
            documents: List of document texts

        Returns:
            Normalized embedding matrix
        """
        embeddings = self.model.encode(
            documents, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
