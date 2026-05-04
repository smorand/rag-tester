"""Tests for the database and embedding factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from rag_tester.providers.databases import (
    ChromaDBProvider,
    SQLiteProvider,
    get_database_provider,
)
from rag_tester.providers.embeddings import (
    GeminiProvider,
    LocalEmbeddingProvider,
    OpenRouterProvider,
    get_embedding_provider,
)


class TestDatabaseFactory:
    def test_resolves_sqlite(self, tmp_path) -> None:
        prov = get_database_provider(f"sqlite:///{tmp_path}/db.sqlite/tbl")
        assert isinstance(prov, SQLiteProvider)

    def test_resolves_chromadb_persistent(self, tmp_path) -> None:
        with patch("rag_tester.providers.databases.chromadb.chromadb") as ch:
            ch.PersistentClient.return_value = MagicMock()
            prov = get_database_provider(f"chromadb://{tmp_path}/coll")
            assert isinstance(prov, ChromaDBProvider)

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported database scheme"):
            get_database_provider("redis://localhost:6379/coll")

    def test_no_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported database scheme"):
            get_database_provider("plain-string-no-scheme")


class TestEmbeddingFactory:
    def test_resolves_gemini(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from rag_tester.config import get_settings

        get_settings.cache_clear()
        prov = get_embedding_provider("gemini", "models/text-embedding-004")
        assert isinstance(prov, GeminiProvider)

    def test_resolves_openrouter(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from rag_tester.config import get_settings

        get_settings.cache_clear()
        prov = get_embedding_provider("openrouter", "openai/text-embedding-3-small")
        assert isinstance(prov, OpenRouterProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            get_embedding_provider("voyage-ai", "voyage-2")

    def test_local_class_is_registered(self) -> None:
        # Don't actually instantiate LocalEmbeddingProvider (loads torch).
        from rag_tester.providers.embeddings import _REGISTRY

        assert _REGISTRY["local"] is LocalEmbeddingProvider
