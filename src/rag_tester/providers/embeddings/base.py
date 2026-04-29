"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different embedding models and APIs.
    """

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, where each vector is a list of floats.
            The length of the outer list matches len(texts).
            The length of each inner list matches get_dimension().

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension for this provider.

        Returns:
            The number of dimensions in each embedding vector
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier for this provider.

        Returns:
            A string identifying the model (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        """
        pass


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""

    pass


class ModelLoadError(EmbeddingError):
    """Raised when a model fails to load."""

    pass
