"""Synthetic document and query generator for testing retrieval quality."""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentQueryPair:
    """A document-query pair with relevance information."""
    doc_id: str
    document: str
    query: str
    relevance: float
    category: str
    semantic_type: str


@dataclass
class SyntheticDataset:
    """A synthetic dataset for retrieval evaluation."""
    documents: List[str]
    document_ids: List[str]
    queries: List[str]
    relevance_map: Dict[str, Dict[str, float]]
    pairs: List[DocumentQueryPair] = field(default_factory=list)

    def get_relevant_docs(self, query_idx: int) -> List[str]:
        """Get relevant document IDs for a query."""
        query_key = f"q_{query_idx}"
        if query_key not in self.relevance_map:
            return []
        return [doc_id for doc_id, rel in self.relevance_map[query_key].items() if rel > 0]


class SyntheticDocumentGenerator:
    """Generator for synthetic documents and queries with semantic variations."""

    TOPICS = {
        "technology": {
            "subtopics": ["artificial intelligence", "cloud computing", "cybersecurity", "blockchain", "IoT"],
            "templates": [
                "The field of {subtopic} has revolutionized how businesses operate in the modern era.",
                "Recent advances in {subtopic} have enabled new applications across industries.",
                "{subtopic} technology continues to evolve, offering improved performance and capabilities.",
                "Organizations are increasingly adopting {subtopic} solutions to enhance their operations.",
                "The future of {subtopic} looks promising with ongoing research and development."
            ],
            "query_templates": [
                "What are the benefits of {subtopic}?",
                "How does {subtopic} work?",
                "Latest developments in {subtopic}",
                "Applications of {subtopic} in business",
                "{subtopic} best practices"
            ]
        },
        "science": {
            "subtopics": ["quantum physics", "genetics", "climate science", "neuroscience", "astronomy"],
            "templates": [
                "Research in {subtopic} has led to groundbreaking discoveries about our universe.",
                "Scientists studying {subtopic} have made significant progress in understanding complex phenomena.",
                "The study of {subtopic} combines theoretical frameworks with experimental validation.",
                "New findings in {subtopic} challenge our previous understanding of natural processes.",
                "{subtopic} research continues to push the boundaries of human knowledge."
            ],
            "query_templates": [
                "Recent discoveries in {subtopic}",
                "How is {subtopic} studied?",
                "Key concepts in {subtopic}",
                "Future of {subtopic} research",
                "{subtopic} breakthroughs"
            ]
        },
        "business": {
            "subtopics": ["marketing strategy", "financial planning", "supply chain", "human resources", "entrepreneurship"],
            "templates": [
                "Effective {subtopic} is essential for organizational success in competitive markets.",
                "Companies that excel at {subtopic} often outperform their competitors.",
                "Modern approaches to {subtopic} leverage data analytics and technology.",
                "Best practices in {subtopic} have evolved significantly over the past decade.",
                "Strategic {subtopic} requires careful planning and execution."
            ],
            "query_templates": [
                "How to improve {subtopic}?",
                "{subtopic} strategies for growth",
                "Common {subtopic} challenges",
                "{subtopic} trends",
                "Best practices for {subtopic}"
            ]
        },
        "health": {
            "subtopics": ["nutrition", "mental health", "exercise science", "preventive medicine", "sleep health"],
            "templates": [
                "Understanding {subtopic} is crucial for maintaining overall well-being.",
                "Research shows that proper attention to {subtopic} can significantly improve quality of life.",
                "Healthcare professionals emphasize the importance of {subtopic} in disease prevention.",
                "New studies in {subtopic} reveal important insights for healthy living.",
                "The connection between {subtopic} and long-term health outcomes is well established."
            ],
            "query_templates": [
                "Benefits of good {subtopic}",
                "How to improve {subtopic}",
                "{subtopic} recommendations",
                "Impact of {subtopic} on health",
                "{subtopic} tips and advice"
            ]
        },
        "education": {
            "subtopics": ["online learning", "STEM education", "early childhood education", "higher education", "vocational training"],
            "templates": [
                "{subtopic} has transformed how knowledge is acquired and shared globally.",
                "Innovations in {subtopic} are making education more accessible than ever before.",
                "The effectiveness of {subtopic} depends on proper implementation and support.",
                "Research on {subtopic} highlights key factors for successful learning outcomes.",
                "{subtopic} continues to adapt to changing societal needs and technological capabilities."
            ],
            "query_templates": [
                "Advantages of {subtopic}",
                "How to succeed in {subtopic}",
                "{subtopic} methods and approaches",
                "Future of {subtopic}",
                "{subtopic} challenges and solutions"
            ]
        }
    }

    SEMANTIC_VARIATIONS = {
        "paraphrase": [
            ("important", "crucial"),
            ("significant", "substantial"),
            ("improve", "enhance"),
            ("new", "novel"),
            ("research", "studies"),
            ("understanding", "comprehension"),
            ("effective", "efficient"),
            ("modern", "contemporary"),
            ("essential", "vital"),
            ("approach", "method")
        ],
        "expansion": [
            ("technology", "technology and innovation"),
            ("research", "research and development"),
            ("business", "business and commerce"),
            ("health", "health and wellness"),
            ("education", "education and learning")
        ],
        "specificity": [
            ("companies", "Fortune 500 companies"),
            ("scientists", "leading researchers"),
            ("studies", "peer-reviewed studies"),
            ("experts", "industry experts"),
            ("organizations", "global organizations")
        ]
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.generated_pairs: List[DocumentQueryPair] = []

    def _apply_semantic_variation(self, text: str, variation_type: str) -> str:
        """Apply semantic variations to text.

        Args:
            text: Original text.
            variation_type: Type of variation to apply.

        Returns:
            Modified text with semantic variations.
        """
        variations = self.SEMANTIC_VARIATIONS.get(variation_type, [])
        modified = text
        for original, replacement in variations:
            if random.random() > 0.5:
                modified = modified.replace(original, replacement)
        return modified

    def _generate_document(self, topic: str, subtopic: str, template_idx: int) -> str:
        """Generate a single document.

        Args:
            topic: Main topic category.
            subtopic: Specific subtopic.
            template_idx: Index of template to use.

        Returns:
            Generated document text.
        """
        templates = self.TOPICS[topic]["templates"]
        template = templates[template_idx % len(templates)]
        document = template.format(subtopic=subtopic)

        variation_type = random.choice(list(self.SEMANTIC_VARIATIONS.keys()))
        document = self._apply_semantic_variation(document, variation_type)

        return document

    def _generate_query(self, topic: str, subtopic: str, template_idx: int) -> str:
        """Generate a query for a document.

        Args:
            topic: Main topic category.
            subtopic: Specific subtopic.
            template_idx: Index of template to use.

        Returns:
            Generated query text.
        """
        query_templates = self.TOPICS[topic]["query_templates"]
        template = query_templates[template_idx % len(query_templates)]
        return template.format(subtopic=subtopic)

    def generate_dataset(
        self,
        num_pairs: int = 100,
        docs_per_query: int = 5,
        include_hard_negatives: bool = True
    ) -> SyntheticDataset:
        """Generate a synthetic dataset for retrieval evaluation.

        Args:
            num_pairs: Number of document-query pairs to generate.
            docs_per_query: Number of documents per query (including negatives).
            include_hard_negatives: Whether to include semantically similar but irrelevant docs.

        Returns:
            SyntheticDataset with documents, queries, and relevance information.
        """
        documents: List[str] = []
        document_ids: List[str] = []
        queries: List[str] = []
        relevance_map: Dict[str, Dict[str, float]] = {}
        pairs: List[DocumentQueryPair] = []

        topics = list(self.TOPICS.keys())
        doc_counter = 0
        query_counter = 0

        pairs_per_topic = num_pairs // len(topics)
        remaining = num_pairs % len(topics)

        for topic_idx, topic in enumerate(topics):
            topic_data = self.TOPICS[topic]
            subtopics = topic_data["subtopics"]

            topic_pairs = pairs_per_topic + (1 if topic_idx < remaining else 0)

            for i in range(topic_pairs):
                subtopic = subtopics[i % len(subtopics)]

                doc_text = self._generate_document(topic, subtopic, i)
                doc_id = f"doc_{doc_counter}"
                documents.append(doc_text)
                document_ids.append(doc_id)

                query_text = self._generate_query(topic, subtopic, i)
                query_key = f"q_{query_counter}"
                queries.append(query_text)

                relevance_map[query_key] = {doc_id: 1.0}

                pair = DocumentQueryPair(
                    doc_id=doc_id,
                    document=doc_text,
                    query=query_text,
                    relevance=1.0,
                    category=topic,
                    semantic_type="exact_match"
                )
                pairs.append(pair)

                if include_hard_negatives:
                    other_subtopics = [s for s in subtopics if s != subtopic]
                    if other_subtopics:
                        neg_subtopic = random.choice(other_subtopics)
                        neg_doc = self._generate_document(topic, neg_subtopic, i + 1)
                        neg_doc_id = f"doc_{doc_counter + 1}"
                        documents.append(neg_doc)
                        document_ids.append(neg_doc_id)
                        relevance_map[query_key][neg_doc_id] = 0.0
                        doc_counter += 1

                        neg_pair = DocumentQueryPair(
                            doc_id=neg_doc_id,
                            document=neg_doc,
                            query=query_text,
                            relevance=0.0,
                            category=topic,
                            semantic_type="hard_negative"
                        )
                        pairs.append(neg_pair)

                doc_counter += 1
                query_counter += 1

        for i in range(min(20, num_pairs // 5)):
            topic = random.choice(topics)
            subtopic = random.choice(self.TOPICS[topic]["subtopics"])

            var_doc = self._generate_document(topic, subtopic, i)
            var_doc = self._apply_semantic_variation(var_doc, "paraphrase")
            var_doc_id = f"doc_{doc_counter}"
            documents.append(var_doc)
            document_ids.append(var_doc_id)

            var_query = self._generate_query(topic, subtopic, i + 1)
            var_query_key = f"q_{query_counter}"
            queries.append(var_query)

            relevance_map[var_query_key] = {var_doc_id: 0.8}

            pair = DocumentQueryPair(
                doc_id=var_doc_id,
                document=var_doc,
                query=var_query,
                relevance=0.8,
                category=topic,
                semantic_type="paraphrase"
            )
            pairs.append(pair)

            doc_counter += 1
            query_counter += 1

        self.generated_pairs = pairs

        return SyntheticDataset(
            documents=documents,
            document_ids=document_ids,
            queries=queries,
            relevance_map=relevance_map,
            pairs=pairs
        )

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about generated data.

        Returns:
            Dictionary with generation statistics.
        """
        if not self.generated_pairs:
            return {"total_pairs": 0}

        stats = {
            "total_pairs": len(self.generated_pairs),
            "categories": {},
            "semantic_types": {}
        }

        for pair in self.generated_pairs:
            stats["categories"][pair.category] = stats["categories"].get(pair.category, 0) + 1
            stats["semantic_types"][pair.semantic_type] = stats["semantic_types"].get(pair.semantic_type, 0) + 1

        return stats
