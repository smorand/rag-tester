"""Vector database provider implementations.

Provides a factory ``get_database_provider`` that resolves a connection-string
URI scheme to its concrete ``VectorDatabase`` subclass. New backends register
themselves in the ``_REGISTRY`` mapping.
"""

from __future__ import annotations

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
    VectorDatabase,
)
from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.databases.elasticsearch import ElasticsearchProvider
from rag_tester.providers.databases.milvus import MilvusProvider
from rag_tester.providers.databases.postgresql import PostgreSQLProvider
from rag_tester.providers.databases.sqlite import SQLiteProvider

_REGISTRY: dict[str, type[VectorDatabase]] = {
    "chromadb": ChromaDBProvider,
    "postgresql": PostgreSQLProvider,
    "milvus": MilvusProvider,
    "sqlite": SQLiteProvider,
    "elasticsearch": ElasticsearchProvider,
}


def get_database_provider(connection_string: str) -> VectorDatabase:
    """Instantiate the database provider matching the URI scheme.

    Args:
        connection_string: Full connection string (e.g.
            ``chromadb://localhost:8000/coll`` or
            ``postgresql://u:p@h:5432/db/tbl``).

    Returns:
        An initialised :class:`VectorDatabase` subclass instance.

    Raises:
        ValueError: If the URI scheme is not registered.
    """
    scheme = connection_string.split("://", 1)[0] if "://" in connection_string else ""
    provider_cls = _REGISTRY.get(scheme)
    if provider_cls is None:
        msg = f"Unsupported database scheme '{scheme}'. Known schemes: {sorted(_REGISTRY.keys())}"
        raise ValueError(msg)
    return provider_cls(connection_string)  # type: ignore[call-arg]


__all__ = [
    "ChromaDBProvider",
    "ConnectionError",
    "DatabaseError",
    "DimensionMismatchError",
    "ElasticsearchProvider",
    "MilvusProvider",
    "PostgreSQLProvider",
    "SQLiteProvider",
    "VectorDatabase",
    "get_database_provider",
]
