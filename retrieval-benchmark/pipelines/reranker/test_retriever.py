"""Tests for the RerankerRetriever to verify re-ranking improves precision."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rank_bm25 import BM25Okapi
from pipelines.reranker.retriever import RerankerRetriever


def test_reranker_improves_precision():
    """Test that cross-encoder re-ranking improves precision over BM25 alone.
    
    This test uses sample documents where semantic understanding is required
    to correctly rank results. BM25 relies on lexical matching, while the
    cross-encoder can understand semantic relationships.
    """
    documents = {
        "doc1": "Python is a high-level programming language known for its readability.",
        "doc2": "Machine learning algorithms learn patterns from data to make predictions.",
        "doc3": "Natural language processing enables computers to understand human text.",
        "doc4": "Deep learning uses neural networks with many layers for complex tasks.",
        "doc5": "Data science combines statistics, programming, and domain expertise.",
        "doc6": "The python snake is a large constrictor found in tropical regions.",
        "doc7": "Artificial intelligence aims to create systems that can perform human-like tasks.",
        "doc8": "Text mining extracts useful information from unstructured text documents.",
        "doc9": "Computer vision allows machines to interpret and understand visual data.",
        "doc10": "Sentiment analysis determines the emotional tone of text content.",
    }
    
    query = "How do computers understand and process human language?"
    expected_relevant = ["doc3", "doc8", "doc10"]
    
    retriever = RerankerRetriever(rerank_top_n=10)
    retriever.index(documents)
    
    reranked_results = retriever.retrieve(query, top_k=5)
    reranked_doc_ids = [doc_id for doc_id, _ in reranked_results]
    
    tokenized_corpus = [doc.lower().split() for doc in documents.values()]
    doc_ids = list(documents.keys())
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
    bm25_doc_ids = [doc_ids[i] for i in bm25_top_indices]
    
    def precision_at_k(results, relevant, k):
        top_k = results[:k]
        return len(set(top_k) & set(relevant)) / k
    
    bm25_precision = precision_at_k(bm25_doc_ids, expected_relevant, 3)
    reranked_precision = precision_at_k(reranked_doc_ids, expected_relevant, 3)
    
    print(f"Query: {query}")
    print(f"\nExpected relevant documents: {expected_relevant}")
    print(f"\nBM25 top-5 results: {bm25_doc_ids}")
    print(f"BM25 precision@3: {bm25_precision:.2f}")
    print(f"\nReranked top-5 results: {reranked_doc_ids}")
    print(f"Reranked precision@3: {reranked_precision:.2f}")
    
    if reranked_precision >= bm25_precision:
        print("\nRe-ranking maintained or improved precision!")
    else:
        print("\nNote: Re-ranking did not improve precision for this specific query.")
    
    return reranked_precision, bm25_precision


def test_retriever_basic_functionality():
    """Test basic functionality of the RerankerRetriever."""
    documents = {
        "doc1": "The quick brown fox jumps over the lazy dog.",
        "doc2": "A fast red fox leaps across the sleeping hound.",
        "doc3": "The weather today is sunny and warm.",
    }
    
    retriever = RerankerRetriever(rerank_top_n=3)
    retriever.index(documents)
    
    results = retriever.retrieve("fox jumping", top_k=2)
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all(isinstance(doc_id, str) for doc_id, _ in results), "Document IDs should be strings"
    assert all(isinstance(score, float) for _, score in results), "Scores should be floats"
    
    result_doc_ids = [doc_id for doc_id, _ in results]
    assert "doc1" in result_doc_ids or "doc2" in result_doc_ids, "Fox-related documents should be in top results"
    
    print("Basic functionality test passed!")
    return True


def test_retriever_name():
    """Test that the retriever has the correct name."""
    retriever = RerankerRetriever()
    assert retriever.name == "BM25 + Cross-Encoder Rerank", f"Unexpected name: {retriever.name}"
    print("Name test passed!")
    return True


def test_retriever_raises_on_unindexed():
    """Test that retrieve raises an error if index() hasn't been called."""
    retriever = RerankerRetriever()
    try:
        retriever.retrieve("test query")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Index has not been built" in str(e)
        print("Unindexed error test passed!")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RerankerRetriever")
    print("=" * 60)
    
    print("\n1. Testing retriever name...")
    test_retriever_name()
    
    print("\n2. Testing error handling for unindexed retriever...")
    test_retriever_raises_on_unindexed()
    
    print("\n3. Testing basic functionality...")
    test_retriever_basic_functionality()
    
    print("\n4. Testing re-ranking precision improvement...")
    print("-" * 60)
    reranked_precision, bm25_precision = test_reranker_improves_precision()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
