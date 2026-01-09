# BM25 + Cross-Encoder Reranker Pipeline

A two-stage retrieval pipeline that combines the efficiency of BM25 for initial candidate retrieval with the precision of cross-encoder models for re-ranking.

## Overview

This pipeline implements a two-stage retrieval approach that balances efficiency and effectiveness. The first stage uses BM25, a fast lexical matching algorithm, to retrieve a larger set of candidate documents. The second stage then applies a cross-encoder model to re-rank these candidates based on semantic similarity, producing more accurate final results.

## Two-Stage Approach

### Stage 1: BM25 Initial Retrieval

BM25 (Best Matching 25) is a probabilistic ranking function that scores documents based on term frequency and inverse document frequency. It excels at fast lexical matching and can efficiently process large document collections. In this pipeline, BM25 retrieves the top 50 candidates (configurable via `rerank_top_n`) for each query.

BM25 is chosen for the first stage because it provides a good balance of speed and recall. While it may miss some semantically relevant documents that don't share exact terms with the query, it efficiently narrows down the search space for the more computationally expensive second stage.

### Stage 2: Cross-Encoder Re-ranking

The cross-encoder model processes each (query, document) pair together, allowing it to capture fine-grained semantic relationships between the query and document. Unlike bi-encoders that encode queries and documents separately, cross-encoders can attend to both inputs simultaneously, resulting in more accurate relevance scores.

The default model `cross-encoder/ms-marco-MiniLM-L-6-v2` is chosen for its excellent balance of speed and effectiveness. It was trained on the MS MARCO passage ranking dataset and provides strong performance on general-domain retrieval tasks while being small enough for practical use.

## Model Choice

The `cross-encoder/ms-marco-MiniLM-L-6-v2` model offers several advantages. It is fast, with only 6 transformer layers making it suitable for re-ranking moderate numbers of candidates. It is effective, having been trained on MS MARCO which is one of the largest passage ranking datasets. It is also general-purpose, performing well across various domains without fine-tuning.

For domain-specific applications, you can substitute a different cross-encoder model by passing the `model_name` parameter to the constructor.

## Usage

```python
from retrieval_benchmark.pipelines.reranker import RerankerRetriever

retriever = RerankerRetriever(
    rerank_top_n=50,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

documents = {
    "doc1": "Python is a programming language.",
    "doc2": "Machine learning uses algorithms to learn from data.",
    "doc3": "Natural language processing deals with text analysis.",
}

retriever.index(documents)

results = retriever.retrieve("What is NLP?", top_k=2)
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

## Configuration

The `RerankerRetriever` accepts the following parameters:

The `rerank_top_n` parameter (default: 50) specifies the number of candidates to retrieve from BM25 for re-ranking. Higher values increase recall but also increase computation time for the cross-encoder stage.

The `model_name` parameter (default: "cross-encoder/ms-marco-MiniLM-L-6-v2") specifies the Hugging Face model identifier for the cross-encoder. You can use any compatible cross-encoder model from the sentence-transformers library.

## Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `rank-bm25`: BM25 implementation for the first stage
- `sentence-transformers`: Cross-encoder models for the second stage

## Performance Considerations

The two-stage approach provides significant benefits over using either method alone. BM25 alone is fast but may miss semantically relevant documents. Cross-encoder alone is accurate but too slow for large collections. The combination leverages BM25's speed for initial filtering and cross-encoder's accuracy for final ranking.

For optimal performance, tune the `rerank_top_n` parameter based on your use case. Higher values improve recall but increase latency. A value of 50-100 typically provides a good balance for most applications.
