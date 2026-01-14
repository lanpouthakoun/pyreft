"""
BM25 Retriever Component

Implements BM25 (Best Matching 25) algorithm for keyword-based document retrieval.
"""

import re
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25-based document retriever for keyword matching."""

    def __init__(self, documents: Optional[List[str]] = None):
        """
        Initialize the BM25 retriever.

        Args:
            documents: Optional list of documents to index
        """
        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if documents:
            self.index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index(self, documents: List[str]) -> None:
        """
        Index documents for BM25 retrieval.

        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

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
        if self.bm25 is None:
            raise ValueError("No documents indexed. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scored_docs[:top_k]:
            results.append((doc_idx, score, self.documents[doc_idx]))

        return results

    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for all documents.

        Args:
            query: Search query

        Returns:
            List of scores for each document
        """
        if self.bm25 is None:
            raise ValueError("No documents indexed. Call index() first.")

        tokenized_query = self._tokenize(query)
        return list(self.bm25.get_scores(tokenized_query))
