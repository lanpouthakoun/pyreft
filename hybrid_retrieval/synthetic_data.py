"""
Synthetic Document and Query Generator

Generates synthetic documents and queries for testing hybrid retrieval systems.
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QueryDocumentPair:
    """A query with its relevant documents."""
    query: str
    relevant_doc_indices: List[int]
    relevance_scores: Dict[int, int]


@dataclass
class SyntheticDataset:
    """A synthetic dataset for retrieval evaluation."""
    documents: List[str]
    queries: List[str]
    relevance_labels: List[Dict[int, int]]
    query_doc_pairs: List[QueryDocumentPair]


class SyntheticDataGenerator:
    """
    Generator for synthetic documents and queries.

    Creates diverse documents across multiple topics with corresponding
    queries that have known relevant documents for evaluation.
    """

    TOPICS = {
        "machine_learning": {
            "keywords": [
                "neural network", "deep learning", "gradient descent",
                "backpropagation", "convolutional", "recurrent", "transformer",
                "attention mechanism", "embedding", "feature extraction",
                "classification", "regression", "clustering", "dimensionality reduction",
                "overfitting", "regularization", "dropout", "batch normalization",
            ],
            "templates": [
                "The {keyword} algorithm is widely used in {application}.",
                "{keyword} has revolutionized the field of {application}.",
                "Recent advances in {keyword} have improved {application} performance.",
                "Researchers are exploring {keyword} for better {application} results.",
                "The combination of {keyword} and {keyword2} enables advanced {application}.",
            ],
            "applications": [
                "image recognition", "natural language processing", "speech recognition",
                "recommendation systems", "autonomous vehicles", "medical diagnosis",
                "fraud detection", "sentiment analysis", "object detection",
            ],
        },
        "databases": {
            "keywords": [
                "SQL", "NoSQL", "relational database", "document store",
                "key-value store", "graph database", "indexing", "query optimization",
                "ACID transactions", "eventual consistency", "sharding", "replication",
                "normalization", "denormalization", "stored procedures", "triggers",
            ],
            "templates": [
                "{keyword} provides efficient data storage for {application}.",
                "Modern {keyword} systems support {application} workloads.",
                "The use of {keyword} improves {application} scalability.",
                "{keyword} is essential for {application} in enterprise systems.",
                "Combining {keyword} with {keyword2} optimizes {application}.",
            ],
            "applications": [
                "e-commerce platforms", "financial systems", "content management",
                "analytics pipelines", "real-time applications", "IoT data storage",
                "social media platforms", "inventory management", "user authentication",
            ],
        },
        "web_development": {
            "keywords": [
                "REST API", "GraphQL", "microservices", "serverless",
                "containerization", "Docker", "Kubernetes", "load balancing",
                "caching", "CDN", "authentication", "authorization",
                "responsive design", "progressive web apps", "single page applications",
            ],
            "templates": [
                "{keyword} architecture enables scalable {application}.",
                "Implementing {keyword} improves {application} user experience.",
                "{keyword} is becoming the standard for {application}.",
                "The adoption of {keyword} has transformed {application}.",
                "{keyword} combined with {keyword2} powers modern {application}.",
            ],
            "applications": [
                "web applications", "mobile backends", "API gateways",
                "content delivery", "user interfaces", "real-time messaging",
                "e-commerce sites", "streaming services", "collaboration tools",
            ],
        },
        "cybersecurity": {
            "keywords": [
                "encryption", "firewall", "intrusion detection", "vulnerability scanning",
                "penetration testing", "zero trust", "multi-factor authentication",
                "security audit", "threat modeling", "incident response",
                "malware analysis", "network security", "endpoint protection",
            ],
            "templates": [
                "{keyword} is critical for protecting {application}.",
                "Organizations implement {keyword} to secure {application}.",
                "Advanced {keyword} techniques defend against {application} threats.",
                "{keyword} compliance is required for {application} systems.",
                "The integration of {keyword} and {keyword2} strengthens {application}.",
            ],
            "applications": [
                "enterprise networks", "cloud infrastructure", "financial transactions",
                "healthcare data", "government systems", "critical infrastructure",
                "personal devices", "IoT networks", "supply chain systems",
            ],
        },
        "data_science": {
            "keywords": [
                "data visualization", "statistical analysis", "hypothesis testing",
                "A/B testing", "time series analysis", "anomaly detection",
                "feature engineering", "data cleaning", "ETL pipelines",
                "data warehousing", "business intelligence", "predictive modeling",
            ],
            "templates": [
                "{keyword} enables insights from {application} data.",
                "Data scientists use {keyword} for {application} analysis.",
                "{keyword} techniques reveal patterns in {application}.",
                "Effective {keyword} drives {application} decision making.",
                "Combining {keyword} with {keyword2} enhances {application} outcomes.",
            ],
            "applications": [
                "customer behavior", "market trends", "operational efficiency",
                "risk assessment", "product development", "resource allocation",
                "performance metrics", "user engagement", "revenue optimization",
            ],
        },
        "cloud_computing": {
            "keywords": [
                "AWS", "Azure", "Google Cloud", "infrastructure as code",
                "auto-scaling", "serverless functions", "object storage",
                "virtual machines", "container orchestration", "service mesh",
                "cloud migration", "hybrid cloud", "multi-cloud strategy",
            ],
            "templates": [
                "{keyword} provides scalable infrastructure for {application}.",
                "Enterprises leverage {keyword} for {application} workloads.",
                "{keyword} reduces costs for {application} deployments.",
                "The shift to {keyword} transforms {application} operations.",
                "{keyword} integrated with {keyword2} optimizes {application}.",
            ],
            "applications": [
                "enterprise applications", "development environments", "disaster recovery",
                "big data processing", "machine learning training", "web hosting",
                "database management", "file storage", "collaboration platforms",
            ],
        },
        "software_engineering": {
            "keywords": [
                "agile methodology", "continuous integration", "continuous deployment",
                "test-driven development", "code review", "version control",
                "design patterns", "refactoring", "technical debt",
                "documentation", "API design", "system architecture",
            ],
            "templates": [
                "{keyword} improves software quality for {application}.",
                "Teams adopt {keyword} to accelerate {application} delivery.",
                "{keyword} practices ensure reliable {application} systems.",
                "Implementing {keyword} reduces bugs in {application}.",
                "{keyword} combined with {keyword2} streamlines {application}.",
            ],
            "applications": [
                "enterprise software", "mobile applications", "embedded systems",
                "gaming platforms", "financial software", "healthcare systems",
                "educational tools", "productivity apps", "communication platforms",
            ],
        },
        "networking": {
            "keywords": [
                "TCP/IP", "HTTP/2", "WebSocket", "DNS", "load balancer",
                "reverse proxy", "VPN", "SDN", "network virtualization",
                "bandwidth optimization", "latency reduction", "packet routing",
            ],
            "templates": [
                "{keyword} enables efficient {application} communication.",
                "Modern {keyword} protocols support {application} requirements.",
                "{keyword} optimization improves {application} performance.",
                "The evolution of {keyword} benefits {application} users.",
                "{keyword} with {keyword2} enhances {application} reliability.",
            ],
            "applications": [
                "video streaming", "online gaming", "remote work",
                "IoT connectivity", "cloud services", "real-time collaboration",
                "content distribution", "enterprise networking", "mobile connectivity",
            ],
        },
    }

    QUERY_TEMPLATES = {
        "what_is": [
            "What is {keyword}?",
            "Explain {keyword} in simple terms.",
            "How does {keyword} work?",
            "What are the basics of {keyword}?",
        ],
        "how_to": [
            "How to implement {keyword}?",
            "Best practices for {keyword}",
            "Guide to using {keyword}",
            "Steps to set up {keyword}",
        ],
        "comparison": [
            "{keyword} vs {keyword2}",
            "Difference between {keyword} and {keyword2}",
            "When to use {keyword} over {keyword2}",
            "Comparing {keyword} and {keyword2}",
        ],
        "application": [
            "Using {keyword} for {application}",
            "{keyword} in {application}",
            "How {keyword} improves {application}",
            "Benefits of {keyword} for {application}",
        ],
        "advanced": [
            "Advanced {keyword} techniques",
            "Optimizing {keyword} performance",
            "Scaling {keyword} systems",
            "Troubleshooting {keyword} issues",
        ],
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def _generate_document(self, topic: str, num_sentences: int = 5) -> str:
        """Generate a single document on a given topic."""
        topic_data = self.TOPICS[topic]
        keywords = topic_data["keywords"]
        templates = topic_data["templates"]
        applications = topic_data["applications"]

        sentences = []
        for _ in range(num_sentences):
            template = random.choice(templates)
            keyword = random.choice(keywords)
            keyword2 = random.choice([k for k in keywords if k != keyword])
            application = random.choice(applications)

            sentence = template.format(
                keyword=keyword,
                keyword2=keyword2,
                application=application,
            )
            sentences.append(sentence)

        return " ".join(sentences)

    def _generate_query(
        self, topic: str, doc_keywords: List[str]
    ) -> Tuple[str, str]:
        """Generate a query related to a topic and document keywords."""
        topic_data = self.TOPICS[topic]
        query_type = random.choice(list(self.QUERY_TEMPLATES.keys()))
        template = random.choice(self.QUERY_TEMPLATES[query_type])

        keyword = random.choice(doc_keywords) if doc_keywords else random.choice(
            topic_data["keywords"]
        )
        keyword2 = random.choice(
            [k for k in topic_data["keywords"] if k != keyword]
        )
        application = random.choice(topic_data["applications"])

        query = template.format(
            keyword=keyword,
            keyword2=keyword2,
            application=application,
        )

        return query, query_type

    def _extract_keywords(self, document: str, topic: str) -> List[str]:
        """Extract topic keywords present in a document."""
        topic_keywords = self.TOPICS[topic]["keywords"]
        doc_lower = document.lower()
        return [kw for kw in topic_keywords if kw.lower() in doc_lower]

    def generate(
        self,
        num_documents: int = 200,
        num_queries: int = 100,
        docs_per_topic: Optional[int] = None,
        sentences_per_doc: int = 5,
        relevant_docs_per_query: int = 3,
    ) -> SyntheticDataset:
        """
        Generate a synthetic dataset.

        Args:
            num_documents: Total number of documents to generate
            num_queries: Number of queries to generate
            docs_per_topic: Documents per topic (auto-calculated if None)
            sentences_per_doc: Sentences per document
            relevant_docs_per_query: Number of relevant docs per query

        Returns:
            SyntheticDataset with documents, queries, and relevance labels
        """
        topics = list(self.TOPICS.keys())
        docs_per_topic = docs_per_topic or (num_documents // len(topics))

        documents: List[str] = []
        doc_topics: List[str] = []
        doc_keywords: List[List[str]] = []

        for topic in topics:
            for _ in range(docs_per_topic):
                doc = self._generate_document(topic, sentences_per_doc)
                documents.append(doc)
                doc_topics.append(topic)
                doc_keywords.append(self._extract_keywords(doc, topic))

        while len(documents) < num_documents:
            topic = random.choice(topics)
            doc = self._generate_document(topic, sentences_per_doc)
            documents.append(doc)
            doc_topics.append(topic)
            doc_keywords.append(self._extract_keywords(doc, topic))

        queries: List[str] = []
        relevance_labels: List[Dict[int, int]] = []
        query_doc_pairs: List[QueryDocumentPair] = []

        for _ in range(num_queries):
            topic = random.choice(topics)

            topic_doc_indices = [
                i for i, t in enumerate(doc_topics) if t == topic
            ]

            if len(topic_doc_indices) < relevant_docs_per_query:
                relevant_indices = topic_doc_indices
            else:
                relevant_indices = random.sample(
                    topic_doc_indices, relevant_docs_per_query
                )

            if relevant_indices:
                primary_doc_idx = relevant_indices[0]
                query, query_type = self._generate_query(
                    topic, doc_keywords[primary_doc_idx]
                )
            else:
                query, query_type = self._generate_query(topic, [])

            relevance = {}
            for rank, doc_idx in enumerate(relevant_indices):
                relevance[doc_idx] = max(1, relevant_docs_per_query - rank)

            partially_relevant = [
                i for i in topic_doc_indices if i not in relevant_indices
            ]
            for doc_idx in random.sample(
                partially_relevant, min(2, len(partially_relevant))
            ):
                relevance[doc_idx] = 1

            queries.append(query)
            relevance_labels.append(relevance)
            query_doc_pairs.append(
                QueryDocumentPair(
                    query=query,
                    relevant_doc_indices=relevant_indices,
                    relevance_scores=relevance,
                )
            )

        return SyntheticDataset(
            documents=documents,
            queries=queries,
            relevance_labels=relevance_labels,
            query_doc_pairs=query_doc_pairs,
        )

    def generate_keyword_focused_queries(
        self,
        documents: List[str],
        num_queries: int = 50,
    ) -> Tuple[List[str], List[Dict[int, int]]]:
        """
        Generate queries focused on specific keywords in documents.

        This creates queries that are more likely to benefit from BM25.

        Args:
            documents: List of documents
            num_queries: Number of queries to generate

        Returns:
            Tuple of (queries, relevance_labels)
        """
        queries = []
        relevance_labels = []

        all_keywords = []
        for topic_data in self.TOPICS.values():
            all_keywords.extend(topic_data["keywords"])

        for _ in range(num_queries):
            keyword = random.choice(all_keywords)

            matching_docs = []
            for i, doc in enumerate(documents):
                if keyword.lower() in doc.lower():
                    matching_docs.append(i)

            query_templates = [
                f"What is {keyword}?",
                f"How does {keyword} work?",
                f"Explain {keyword}",
                f"Benefits of {keyword}",
                f"{keyword} best practices",
            ]
            query = random.choice(query_templates)
            queries.append(query)

            relevance = {doc_idx: 2 for doc_idx in matching_docs[:5]}
            relevance_labels.append(relevance)

        return queries, relevance_labels

    def generate_semantic_queries(
        self,
        documents: List[str],
        num_queries: int = 50,
    ) -> Tuple[List[str], List[Dict[int, int]]]:
        """
        Generate queries that require semantic understanding.

        This creates queries that are more likely to benefit from dense retrieval.

        Args:
            documents: List of documents
            num_queries: Number of queries to generate

        Returns:
            Tuple of (queries, relevance_labels)
        """
        queries = []
        relevance_labels = []

        semantic_templates = [
            "How can I improve system performance?",
            "What are the best ways to handle large amounts of data?",
            "How do I make my application more secure?",
            "What techniques help with scaling?",
            "How can I reduce latency in my system?",
            "What are modern approaches to software development?",
            "How do I ensure data consistency?",
            "What methods improve user experience?",
            "How can I automate repetitive tasks?",
            "What strategies help with debugging?",
        ]

        topic_relevance = {
            "improve system performance": ["machine_learning", "databases", "cloud_computing"],
            "handle large amounts of data": ["databases", "data_science", "cloud_computing"],
            "application more secure": ["cybersecurity", "web_development"],
            "techniques help with scaling": ["cloud_computing", "databases", "web_development"],
            "reduce latency": ["networking", "cloud_computing", "databases"],
            "modern approaches to software": ["software_engineering", "web_development"],
            "ensure data consistency": ["databases", "software_engineering"],
            "improve user experience": ["web_development", "data_science"],
            "automate repetitive tasks": ["software_engineering", "cloud_computing"],
            "strategies help with debugging": ["software_engineering", "data_science"],
        }

        for _ in range(num_queries):
            query = random.choice(semantic_templates)
            queries.append(query)

            relevant_topics = []
            for pattern, topics in topic_relevance.items():
                if pattern in query.lower():
                    relevant_topics = topics
                    break

            if not relevant_topics:
                relevant_topics = list(self.TOPICS.keys())[:3]

            relevance = {}
            for i, doc in enumerate(documents):
                doc_lower = doc.lower()
                score = 0
                for topic in relevant_topics:
                    topic_keywords = self.TOPICS[topic]["keywords"]
                    matches = sum(1 for kw in topic_keywords if kw.lower() in doc_lower)
                    score += matches

                if score > 0:
                    relevance[i] = min(3, score)

            top_relevant = sorted(relevance.items(), key=lambda x: x[1], reverse=True)[:10]
            relevance_labels.append(dict(top_relevant))

        return queries, relevance_labels
