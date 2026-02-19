"""
Moral Context Retriever

RAG-based retrieval system for providing moral framework context
during LLM evaluation.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from phronesis.rag.faiss_index import FAISSIndex, SearchResult

logger = logging.getLogger(__name__)

# Sentence transformers import with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")


@dataclass
class MoralContext:
    """Retrieved moral context for a query."""

    query: str
    contexts: list[str]
    scores: list[float]
    sources: list[str] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format contexts for injection into prompt."""
        if not self.contexts:
            return ""

        formatted = "Relevant moral context:\n"
        for i, (ctx, score) in enumerate(zip(self.contexts, self.scores), 1):
            formatted += f"{i}. {ctx}\n"

        return formatted


@dataclass
class MoralDocument:
    """A document containing moral framework information."""

    text: str
    source: str
    framework: str
    metadata: dict = field(default_factory=dict)


class MoralContextRetriever:
    """
    Retrieval system for moral framework contexts.

    Uses sentence embeddings and FAISS for efficient retrieval of
    relevant moral framework information during evaluation.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[Path] = None,
        dimension: int = 384,
        index_type: str = "flat",
        use_gpu: bool = True,
    ):
        """
        Initialize the retriever.

        Args:
            embedding_model: Model for generating embeddings
            index_path: Path to load existing index
            dimension: Embedding dimension (must match model)
            index_type: FAISS index type
            use_gpu: Whether to use GPU for retrieval
        """
        self.embedding_model_name = embedding_model
        self.dimension = dimension

        # Initialize embedding model
        self._encoder: Optional["SentenceTransformer"] = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._encoder = SentenceTransformer(embedding_model)
            self.dimension = self._encoder.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = FAISSIndex(
            dimension=self.dimension,
            index_type=index_type,
            use_gpu=use_gpu,
        )

        # Document storage
        self._documents: list[MoralDocument] = []

        # Load existing index if provided
        if index_path and Path(index_path).exists():
            self.load(index_path)

        logger.info(f"MoralContextRetriever initialized with dim={self.dimension}")

    def add_documents(
        self,
        documents: list[MoralDocument],
        batch_size: int = 32,
    ) -> None:
        """
        Add documents to the retrieval index.

        Args:
            documents: List of moral framework documents
            batch_size: Batch size for encoding
        """
        texts = [doc.text for doc in documents]
        embeddings = self._encode_texts(texts, batch_size)

        self.index.add(embeddings, texts)
        self._documents.extend(documents)

        logger.info(f"Added {len(documents)} documents to retriever")

    def add_texts(
        self,
        texts: list[str],
        source: str = "custom",
        framework: str = "general",
        batch_size: int = 32,
    ) -> None:
        """
        Add text strings to the retrieval index.

        Args:
            texts: List of text strings
            source: Source identifier
            framework: Associated moral framework
            batch_size: Batch size for encoding
        """
        documents = [
            MoralDocument(text=t, source=source, framework=framework)
            for t in texts
        ]
        self.add_documents(documents, batch_size)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> MoralContext:
        """
        Retrieve relevant moral contexts for a query.

        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            MoralContext with relevant documents
        """
        query_embedding = self.encode_text(query)
        results = self.index.search(query_embedding.reshape(1, -1), k=k)

        contexts = []
        scores = []
        sources = []

        for i in range(len(results.indices[0])):
            idx = results.indices[0][i]
            distance = results.distances[0][i]

            # Convert L2 distance to similarity score
            similarity = 1 / (1 + distance)

            if similarity >= threshold and idx >= 0:
                if results.texts:
                    contexts.append(results.texts[0][i])
                elif idx < len(self._documents):
                    contexts.append(self._documents[idx].text)

                scores.append(float(similarity))

                if idx < len(self._documents):
                    sources.append(self._documents[idx].source)

        return MoralContext(
            query=query,
            contexts=contexts,
            scores=scores,
            sources=sources,
        )

    def retrieve_batch(
        self,
        queries: list[str],
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[MoralContext]:
        """
        Retrieve contexts for multiple queries efficiently.

        Args:
            queries: List of query texts
            k: Number of results per query
            threshold: Minimum similarity threshold

        Returns:
            List of MoralContext objects
        """
        query_embeddings = self._encode_texts(queries)
        results = self.index.search(query_embeddings, k=k)

        contexts = []
        for q_idx, query in enumerate(queries):
            query_contexts = []
            query_scores = []
            query_sources = []

            for i in range(k):
                idx = results.indices[q_idx][i]
                distance = results.distances[q_idx][i]
                similarity = 1 / (1 + distance)

                if similarity >= threshold and idx >= 0:
                    if results.texts:
                        query_contexts.append(results.texts[q_idx][i])
                    elif idx < len(self._documents):
                        query_contexts.append(self._documents[idx].text)

                    query_scores.append(float(similarity))

                    if idx < len(self._documents):
                        query_sources.append(self._documents[idx].source)

            contexts.append(MoralContext(
                query=query,
                contexts=query_contexts,
                scores=query_scores,
                sources=query_sources,
            ))

        return contexts

    def encode_text(self, text: str) -> NDArray[np.float32]:
        """Encode a single text to embedding."""
        if self._encoder is not None:
            return self._encoder.encode(text, convert_to_numpy=True)
        else:
            # Random fallback for testing
            return np.random.randn(self.dimension).astype(np.float32)

    def _encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """Encode multiple texts to embeddings."""
        if self._encoder is not None:
            return self._encoder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
            )
        else:
            # Random fallback for testing
            return np.random.randn(len(texts), self.dimension).astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Save retriever state to disk."""
        path = Path(path)
        self.index.save(path)

        # Save documents
        import json
        docs_data = [
            {
                "text": d.text,
                "source": d.source,
                "framework": d.framework,
                "metadata": d.metadata,
            }
            for d in self._documents
        ]
        with open(path.with_suffix(".docs.json"), "w") as f:
            json.dump(docs_data, f)

        logger.info(f"Retriever saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load retriever state from disk."""
        path = Path(path)
        self.index.load(path)

        # Load documents
        docs_path = path.with_suffix(".docs.json")
        if docs_path.exists():
            import json
            with open(docs_path) as f:
                docs_data = json.load(f)

            self._documents = [
                MoralDocument(**d) for d in docs_data
            ]

        logger.info(f"Retriever loaded from {path}")

    @classmethod
    def with_moral_frameworks(cls, **kwargs) -> "MoralContextRetriever":
        """
        Create retriever pre-populated with moral framework knowledge.

        Returns a retriever with embedded knowledge about major ethical
        frameworks for context augmentation.
        """
        retriever = cls(**kwargs)

        # Add foundational moral framework texts
        framework_texts = [
            # Utilitarianism
            MoralDocument(
                text=(
                    "Utilitarianism holds that the morally right action is "
                    "the one that produces the most good for the greatest "
                    "number of people. Actions are judged solely by their "
                    "consequences."
                ),
                source="ethical_foundations",
                framework="utilitarian",
            ),
            MoralDocument(
                text=(
                    "The principle of utility requires impartial consideration "
                    "of all affected parties. No one's happiness counts more "
                    "than anyone else's."
                ),
                source="ethical_foundations",
                framework="utilitarian",
            ),
            # Deontology
            MoralDocument(
                text=(
                    "Deontological ethics holds that certain actions are "
                    "inherently right or wrong, regardless of their consequences. "
                    "Moral rules are categorical imperatives."
                ),
                source="ethical_foundations",
                framework="deontological",
            ),
            MoralDocument(
                text=(
                    "The categorical imperative: Act only according to maxims "
                    "you could will to become universal laws. Treat humanity "
                    "never merely as means but always as ends."
                ),
                source="ethical_foundations",
                framework="deontological",
            ),
            # Virtue Ethics
            MoralDocument(
                text=(
                    "Virtue ethics focuses on character rather than rules or "
                    "consequences. A virtuous person cultivates excellence "
                    "through practice and practical wisdom (phronesis)."
                ),
                source="ethical_foundations",
                framework="virtue_ethics",
            ),
            # AI Safety
            MoralDocument(
                text=(
                    "AI alignment aims to ensure artificial intelligence "
                    "systems act in accordance with human values and intentions. "
                    "Corrigibility is the property of accepting human oversight."
                ),
                source="ai_safety",
                framework="alignment",
            ),
            MoralDocument(
                text=(
                    "Instrumental convergence suggests that sufficiently "
                    "advanced AI systems may develop convergent goals like "
                    "self-preservation regardless of their terminal objectives."
                ),
                source="ai_safety",
                framework="alignment",
            ),
        ]

        retriever.add_documents(framework_texts)

        return retriever
