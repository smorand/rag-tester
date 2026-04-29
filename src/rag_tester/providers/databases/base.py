"""Abstract base class for vector database providers."""

from abc import ABC, abstractmethod
from typing import Any


class VectorDatabase(ABC):
    """Abstract base class for vector database providers.

    All vector database providers must implement this interface to ensure
    consistent behavior across different database backends.
    """

    @abstractmethod
    async def create_collection(self, name: str, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        """Create a collection/table for embeddings.

        Args:
            name: Name of the collection
            dimension: Embedding dimension (number of floats in each vector)
            metadata: Optional metadata for the collection

        Raises:
            DatabaseError: If collection creation fails
        """
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Name of the collection to check

        Returns:
            True if the collection exists, False otherwise

        Raises:
            DatabaseError: If the check fails
        """
        pass

    @abstractmethod
    async def insert(self, collection: str, records: list[dict[str, Any]]) -> None:
        """Insert records with embeddings into a collection.

        Args:
            collection: Name of the collection
            records: List of records, each containing:
                - id: Unique identifier (str)
                - text: Original text (str)
                - embedding: Embedding vector (list[float])
                - metadata: Optional additional metadata (dict)

        Raises:
            DatabaseError: If insertion fails
            DimensionMismatchError: If embedding dimension doesn't match collection
        """
        pass

    @abstractmethod
    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query for similar embeddings.

        Args:
            collection: Name of the collection to query
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of records sorted by similarity (descending), each containing:
                - id: Document identifier
                - text: Original text
                - score: Similarity score (higher is more similar)
                - metadata: Optional additional metadata

        Raises:
            DatabaseError: If query fails
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data.

        Args:
            name: Name of the collection to delete

        Raises:
            DatabaseError: If deletion fails
        """
        pass

    @abstractmethod
    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            name: Name of the collection

        Returns:
            Dictionary containing collection metadata:
                - name: Collection name
                - dimension: Embedding dimension
                - count: Number of documents
                - metadata: Additional metadata

        Raises:
            DatabaseError: If collection doesn't exist or info retrieval fails
        """
        pass


class DatabaseError(Exception):
    """Base exception for database-related errors."""

    pass


class DimensionMismatchError(DatabaseError):
    """Raised when embedding dimension doesn't match collection dimension."""

    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass
