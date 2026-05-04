"""Local embedding provider using sentence-transformers."""

import logging
from typing import Any

from opentelemetry import trace

from rag_tester.providers.embeddings.base import EmbeddingProvider, ModelLoadError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers models.

    This provider generates embeddings using locally-run sentence-transformers
    models. Models are downloaded on first use and cached for subsequent calls.

    Args:
        model_name: HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        device: Device to run on ("cpu", "cuda", "mps"). Defaults to "cpu".

    Raises:
        ModelLoadError: If the model fails to load
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Initialize the local embedding provider.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu", "cuda", "mps")
        """
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._dimension: int | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformers model.

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Import here to avoid loading heavy dependencies at module level
            from sentence_transformers import SentenceTransformer

            logger.info("Loading model: %s on device: %s", self._model_name, self._device)

            with tracer.start_as_current_span("model_load") as span:
                span.set_attribute("model.name", self._model_name)
                span.set_attribute("model.device", self._device)

                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._dimension = self._model.get_sentence_embedding_dimension()

                span.set_attribute("model.dimension", self._dimension)
                logger.info("Model loaded: %s (dimension: %s)", self._model_name, self._dimension)

        except Exception as e:
            error_msg = f"Failed to load model: {self._model_name}"
            logger.error("%s: %s", error_msg, e)
            raise ModelLoadError(error_msg) from e

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        with tracer.start_as_current_span("embed_texts") as span:
            span.set_attribute("model.name", self._model_name)
            span.set_attribute("texts.count", len(texts))

            try:
                # sentence-transformers encode is synchronous (CPU-bound)
                # We run it directly without asyncio.to_thread since it's already optimized
                embeddings = self._model.encode(texts, convert_to_numpy=True)

                # Convert numpy arrays to lists for JSON serialization
                result = [embedding.tolist() for embedding in embeddings]

                span.set_attribute("embeddings.count", len(result))
                logger.debug("Generated %s embeddings", len(result))

                return result

            except Exception as e:
                logger.error("Failed to generate embeddings: %s", e)
                span.record_exception(e)
                raise

    def get_dimension(self) -> int:
        """Return the embedding dimension for this model.

        Returns:
            The number of dimensions in each embedding vector
        """
        if self._dimension is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
        return self._dimension

    def get_model_name(self) -> str:
        """Return the model identifier.

        Returns:
            The HuggingFace model identifier
        """
        return self._model_name
