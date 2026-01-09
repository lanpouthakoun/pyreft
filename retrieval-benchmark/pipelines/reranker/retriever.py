"""Two-stage retrieval pipeline with BM25 for initial retrieval and cross-encoder for re-ranking."""

from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

try:
    from ...shared.interface import BaseRetriever
except ImportError:
    from shared.interface import BaseRetriever


class RerankerRetriever(BaseRetriever):
    """Two-stage retrieval pipeline using BM25 for initial retrieval and cross-encoder for re-ranking.
    
    This pipeline implements a two-stage approach:
    1. First stage: BM25 retrieves top-N candidates (N > top_k)
    2. Second stage: Cross-encoder scores each (query, document) pair for precise re-ranking
    
    The cross-encoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2' is used by default,
    which provides a good balance between speed and effectiveness.
    """
    
    def __init__(
        self,
        rerank_top_n: int = 50,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize the RerankerRetriever.
        
        Args:
            rerank_top_n: Number of candidates to retrieve from BM25 for re-ranking.
                         Should be larger than the final top_k to allow re-ranking to improve results.
            model_name: Name of the cross-encoder model to use for re-ranking.
        """
        self.rerank_top_n = rerank_top_n
        self.model_name = model_name
        self.cross_encoder = CrossEncoder(model_name)
        self.bm25 = None
        self.documents: Dict[str, str] = {}
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
    
    @property
    def name(self) -> str:
        """Return the name of the retriever."""
        return "BM25 + Cross-Encoder Rerank"
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens.
        """
        return text.lower().split()
    
    def index(self, documents: Dict[str, str]) -> None:
        """Index a collection of documents for retrieval.
        
        This method:
        1. Stores the full document content for re-ranking
        2. Tokenizes documents for BM25 indexing
        3. Builds the BM25 index
        
        Args:
            documents: Dictionary mapping document IDs to document content.
        """
        self.documents = documents
        self.doc_ids = list(documents.keys())
        self.tokenized_corpus = [self._tokenize(documents[doc_id]) for doc_id in self.doc_ids]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents using two-stage retrieval.
        
        Stage 1: BM25 retrieves top-N candidates (N = rerank_top_n)
        Stage 2: Cross-encoder re-ranks candidates and returns top-k
        
        Args:
            query: The search query.
            top_k: Number of documents to return after re-ranking.
            
        Returns:
            List of tuples containing (document_id, score) sorted by relevance.
            
        Raises:
            ValueError: If index() has not been called yet.
        """
        if self.bm25 is None:
            raise ValueError("Index has not been built. Call index() first.")
        
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        num_candidates = min(self.rerank_top_n, len(self.doc_ids))
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:num_candidates]
        
        candidate_doc_ids = [self.doc_ids[i] for i in top_indices]
        candidate_docs = [self.documents[doc_id] for doc_id in candidate_doc_ids]
        
        query_doc_pairs = [[query, doc] for doc in candidate_docs]
        cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
        
        scored_candidates = list(zip(candidate_doc_ids, cross_encoder_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [(doc_id, float(score)) for doc_id, score in scored_candidates[:top_k]]
