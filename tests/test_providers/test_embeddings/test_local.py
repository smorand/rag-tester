"""Tests for LocalEmbeddingProvider."""

import pytest

from rag_tester.providers.embeddings.base import ModelLoadError
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider


class TestLocalEmbeddingProvider:
    """Tests for LocalEmbeddingProvider."""

    @pytest.fixture
    def provider(self) -> LocalEmbeddingProvider:
        """Create a LocalEmbeddingProvider instance with a small model."""
        # Use a small, fast model for testing
        return LocalEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    async def test_embed_texts_success(self, provider: LocalEmbeddingProvider) -> None:
        """Test successful embedding generation."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks.",
            "Python is a programming language.",
        ]

        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 384 for emb in embeddings)  # all-MiniLM-L6-v2 has 384 dimensions
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    async def test_embed_texts_empty_list(self, provider: LocalEmbeddingProvider) -> None:
        """Test embedding empty list returns empty list."""
        embeddings = await provider.embed_texts([])
        assert embeddings == []

    async def test_embed_texts_single_text(self, provider: LocalEmbeddingProvider) -> None:
        """Test embedding a single text."""
        texts = ["Test text"]
        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    async def test_embed_texts_unicode(self, provider: LocalEmbeddingProvider) -> None:
        """Test embedding text with Unicode and emoji."""
        texts = ["Hello 世界 🌍 مرحبا"]
        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    async def test_embed_texts_long_text(self, provider: LocalEmbeddingProvider) -> None:
        """Test embedding very long text (10K chars)."""
        long_text = "a" * 10000
        embeddings = await provider.embed_texts([long_text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_get_dimension(self, provider: LocalEmbeddingProvider) -> None:
        """Test getting embedding dimension."""
        dimension = provider.get_dimension()
        assert dimension == 384

    def test_get_model_name(self, provider: LocalEmbeddingProvider) -> None:
        """Test getting model name."""
        model_name = provider.get_model_name()
        assert model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_invalid_model_name(self) -> None:
        """Test that invalid model name raises ModelLoadError."""
        with pytest.raises(ModelLoadError, match="Failed to load model: invalid/model-name"):
            LocalEmbeddingProvider("invalid/model-name", device="cpu")

    async def test_embeddings_are_consistent(self, provider: LocalEmbeddingProvider) -> None:
        """Test that same text produces same embedding."""
        text = "Test consistency"

        embeddings1 = await provider.embed_texts([text])
        embeddings2 = await provider.embed_texts([text])

        # Embeddings should be identical (deterministic)
        assert embeddings1[0] == embeddings2[0]

    async def test_different_texts_different_embeddings(self, provider: LocalEmbeddingProvider) -> None:
        """Test that different texts produce different embeddings."""
        texts = ["First text", "Second text"]
        embeddings = await provider.embed_texts(texts)

        # Embeddings should be different
        assert embeddings[0] != embeddings[1]

    async def test_similar_texts_similar_embeddings(self, provider: LocalEmbeddingProvider) -> None:
        """Test that similar texts produce similar embeddings."""
        texts = [
            "Machine learning is a subset of AI",
            "ML is part of artificial intelligence",
        ]
        embeddings = await provider.embed_texts(texts)

        # Calculate cosine similarity
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b, strict=False))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(y * y for y in b))
            return dot_product / (magnitude_a * magnitude_b)

        similarity = cosine_similarity(embeddings[0], embeddings[1])

        # Similar texts should have high cosine similarity (> 0.5)
        assert similarity > 0.5
