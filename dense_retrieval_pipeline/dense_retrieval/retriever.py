"""Dense vector retrieval module using sentence-transformers and FAISS."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DenseRetriever:
    """Dense retrieval system using sentence-transformers embeddings and FAISS index."""

    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "description": "Fast, good quality, 384 dimensions",
            "use_case": "General purpose, balanced speed/quality"
        },
        "all-mpnet-base-v2": {
            "dimension": 768,
            "description": "High quality, 768 dimensions",
            "use_case": "When quality is more important than speed"
        },
        "paraphrase-MiniLM-L6-v2": {
            "dimension": 384,
            "description": "Optimized for paraphrase detection",
            "use_case": "Semantic similarity, duplicate detection"
        },
        "multi-qa-MiniLM-L6-cos-v1": {
            "dimension": 384,
            "description": "Trained on Q&A pairs",
            "use_case": "Question answering, FAQ retrieval"
        }
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        nlist: int = 100
    ):
        """Initialize the dense retriever.

        Args:
            model_name: Name of the sentence-transformer model to use.
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw').
            nlist: Number of clusters for IVF index.
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_type = index_type
        self.nlist = nlist
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.document_ids: List[str] = []

    def _create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a FAISS index based on the specified type.

        Args:
            embeddings: Document embeddings to index.

        Returns:
            FAISS index.
        """
        n_vectors = embeddings.shape[0]

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            effective_nlist = min(self.nlist, n_vectors)
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, effective_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def index_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> None:
        """Index a collection of documents.

        Args:
            documents: List of document texts to index.
            document_ids: Optional list of document IDs. If not provided, uses indices.
            batch_size: Batch size for encoding.
        """
        self.documents = documents
        self.document_ids = document_ids if document_ids else [str(i) for i in range(len(documents))]

        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings = embeddings.astype(np.float32)

        self.index = self._create_index(embeddings)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Search for documents similar to the query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of tuples (document_id, document_text, score).
        """
        if self.index is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        if self.index_type == "ivf":
            self.index.nprobe = min(10, self.nlist)

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((
                    self.document_ids[idx],
                    self.documents[idx],
                    float(score)
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
        if self.index is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        query_embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        if self.index_type == "ivf":
            self.index.nprobe = min(10, self.nlist)

        scores, indices = self.index.search(query_embeddings, min(top_k, len(self.documents)))

        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx >= 0:
                    results.append((
                        self.document_ids[idx],
                        self.documents[idx],
                        float(score)
                    ))
            all_results.append(results)

        return all_results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_documents": len(self.documents),
            "supported_models": self.SUPPORTED_MODELS
        }

    @classmethod
    def get_model_recommendations(cls) -> Dict[str, Dict[str, str]]:
        """Get recommendations for model selection.

        Returns:
            Dictionary of model recommendations.
        """
        return cls.SUPPORTED_MODELS
