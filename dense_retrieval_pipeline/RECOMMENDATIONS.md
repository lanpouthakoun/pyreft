# Dense Vector Retrieval: Recommendations and Best Practices

## Embedding Model Selection

### Recommended Models by Use Case

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | General purpose, balanced speed/quality |
| `all-mpnet-base-v2` | 768 | Medium | Excellent | When quality is paramount |
| `paraphrase-MiniLM-L6-v2` | 384 | Fast | Good | Semantic similarity, duplicate detection |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Question answering, FAQ retrieval |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good | Multilingual applications |

### Model Selection Guidelines

For most applications, start with `all-MiniLM-L6-v2` as it provides an excellent balance between speed and quality. This model works well for general semantic search, document retrieval, and similarity matching tasks.

If your application requires the highest possible quality and latency is less critical, consider `all-mpnet-base-v2`. This model produces higher-quality embeddings at the cost of increased computation time and memory usage.

For question-answering systems or FAQ retrieval, `multi-qa-MiniLM-L6-cos-v1` is specifically trained on question-answer pairs and may provide better results for these use cases.

## Index Type Selection

### FAISS Index Types

The choice of index type significantly impacts both search speed and accuracy.

**Flat Index (IndexFlatIP)** provides exact nearest neighbor search with 100% recall. Use this for small datasets (under 10,000 documents) or when accuracy is critical. Query time scales linearly with dataset size.

**IVF Index (IndexIVFFlat)** partitions the vector space into clusters for faster approximate search. Recommended for medium-sized datasets (10,000 to 1,000,000 documents). The `nlist` parameter controls the number of clusters, and `nprobe` controls how many clusters to search at query time. Higher values improve recall at the cost of speed.

**HNSW Index (IndexHNSWFlat)** uses a hierarchical navigable small world graph for efficient approximate search. Best for large datasets (over 1,000,000 documents) where query latency is critical. Provides excellent speed-accuracy tradeoff but requires more memory.

### Index Selection Guidelines

| Dataset Size | Recommended Index | Parameters |
|--------------|-------------------|------------|
| < 10,000 | Flat | N/A |
| 10,000 - 100,000 | IVF | nlist=100, nprobe=10 |
| 100,000 - 1,000,000 | IVF | nlist=1000, nprobe=50 |
| > 1,000,000 | HNSW | M=32, efSearch=64 |

## Optimal Use Cases for Dense Retrieval

### Where Dense Retrieval Excels

Dense retrieval is particularly effective for semantic search where the goal is to find conceptually similar content rather than exact keyword matches. It handles synonyms, paraphrases, and related concepts naturally without requiring explicit query expansion.

Common successful applications include document search and retrieval systems, FAQ and knowledge base search, semantic similarity and duplicate detection, recommendation systems based on content similarity, and cross-lingual retrieval when using multilingual models.

### Where to Consider Alternatives

Dense retrieval may underperform for exact keyword matching requirements, highly technical domains with specialized vocabulary not well-represented in pre-trained models, very short queries (single words), and cases where interpretability of results is critical.

For these scenarios, consider hybrid approaches that combine dense retrieval with traditional sparse methods like BM25. Hybrid retrieval can capture both semantic similarity and exact keyword matches.

## Performance Optimization

### Batch Processing

When indexing large document collections, use batch encoding with appropriate batch sizes (typically 32-128) to maximize GPU utilization if available. For CPU-only environments, smaller batch sizes may be more efficient.

### Query Optimization

For applications with high query throughput, consider pre-computing query embeddings for common queries, using batch search for multiple simultaneous queries, and implementing caching for frequently accessed results.

### Memory Management

Embedding models and FAISS indexes can consume significant memory. For large-scale deployments, consider using memory-mapped indexes, implementing index sharding across multiple machines, and using quantized indexes (e.g., IndexIVFPQ) for reduced memory footprint.

## Evaluation Metrics Interpretation

### Key Metrics

**MRR (Mean Reciprocal Rank)** measures how quickly the first relevant result appears. Values above 0.8 indicate excellent performance for single-answer retrieval tasks.

**Precision@K** measures the proportion of relevant documents in the top-K results. High Precision@1 is critical for applications showing only the top result.

**Recall@K** measures the proportion of all relevant documents found in the top-K results. Important when finding all relevant documents matters more than ranking.

**NDCG@K** considers both relevance and ranking position. Useful when you have graded relevance judgments rather than binary relevance.

### Benchmark Targets

For general-purpose semantic search, aim for MRR above 0.7, Precision@1 above 0.6, Recall@10 above 0.8, and NDCG@10 above 0.7.

These targets may vary based on your specific domain and requirements. Always evaluate on data representative of your actual use case.
