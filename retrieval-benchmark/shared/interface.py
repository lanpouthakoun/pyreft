"""Base interface for retrieval pipelines."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseRetriever(ABC):
    """Abstract base class for retrieval pipelines.
    
    All retrieval implementations should inherit from this class
    and implement the required methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the retriever."""
        pass
    
    @abstractmethod
    def index(self, documents: Dict[str, str]) -> None:
        """Index a collection of documents.
        
        Args:
            documents: Dictionary mapping document IDs to document content.
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The search query.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of tuples containing (document_id, score) sorted by relevance.
        """
        pass
