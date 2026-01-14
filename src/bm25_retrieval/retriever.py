"""BM25 sparse retrieval module using rank_bm25."""

import re
from typing import List, Tuple, Optional, Dict, Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25Retriever:
    """BM25 sparse retrieval system using rank_bm25."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """Initialize the BM25 retriever.

        Args:
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            epsilon: BM25 floor value for IDF.
        """
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required. Install with: pip install rank-bm25")

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.document_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

    def _preprocess(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing/querying.

        Applies lowercasing, removes punctuation, and tokenizes on whitespace.

        Args:
            text: Input text to preprocess.

        Returns:
            List of preprocessed tokens.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        return tokens

    def index_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None
    ) -> None:
        """Index a collection of documents.

        Args:
            documents: List of document texts to index.
            document_ids: Optional list of document IDs. If not provided, uses indices.
        """
        self.documents = documents
        self.document_ids = document_ids if document_ids else [str(i) for i in range(len(documents))]

        self.tokenized_corpus = [self._preprocess(doc) for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Search for documents matching the query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of tuples (document_id, document_text, score).
        """
        if self.bm25 is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        tokenized_query = self._preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self.document_ids[idx],
                self.documents[idx],
                float(scores[idx])
            ))

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[str, str, float]]]:
        """Search for multiple queries.

        Args:
            queries: List of query texts.
            top_k: Number of results per query.

        Returns:
            List of results for each query.
        """
        return [self.search(query, top_k) for query in queries]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": "BM25Okapi",
            "model_type": "sparse",
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "num_documents": len(self.documents),
            "avg_doc_length": sum(len(tokens) for tokens in self.tokenized_corpus) / len(self.tokenized_corpus) if self.tokenized_corpus else 0,
            "vocabulary_size": len(set(token for tokens in self.tokenized_corpus for token in tokens)) if self.tokenized_corpus else 0
        }
