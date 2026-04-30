"""Core testing logic for RAG Tester.

This module provides the Tester class for executing queries against vector databases
and formatting results in multiple output formats.
"""

import json
import logging
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from rag_tester.providers.databases.base import DatabaseError, VectorDatabase
from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_tester.tracing import trace_span

logger = logging.getLogger(__name__)


class TestError(Exception):
    """Base exception for testing-related errors."""

    pass


class ValidationError(TestError):
    """Raised when input validation fails."""

    pass


class Tester:
    """Handles query testing against vector databases.

    This class provides methods for executing queries, validating inputs,
    and formatting results in multiple output formats (table, JSON, text).
    """

    def __init__(
        self,
        database: VectorDatabase,
        embedding_provider: EmbeddingProvider,
        collection_name: str,
    ) -> None:
        """Initialize the Tester.

        Args:
            database: Vector database provider
            embedding_provider: Embedding provider for query encoding
            collection_name: Name of the collection to query
        """
        self.database = database
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name

    async def test_query(
        self,
        query: str,
        top_k: int = 5,
        output_format: str = "table",
    ) -> dict[str, Any]:
        """Execute a test query and return formatted results.

        Args:
            query: Natural language query text
            top_k: Number of results to return (1-100)
            output_format: Output format (table, json, text)

        Returns:
            Dictionary containing:
                - query: Original query text
                - results: List of result dictionaries
                - tokens: Token count (0 for local models)
                - time: Execution time in seconds

        Raises:
            ValidationError: If inputs are invalid
            TestError: If query execution fails
        """
        # Validate inputs
        self._validate_query(query)
        self._validate_top_k(top_k)
        self._validate_format(output_format)

        start_time = time.time()

        try:
            # Check if collection exists and is not empty
            with trace_span("tester.validate_collection"):
                if not await self.database.collection_exists(self.collection_name):
                    raise TestError(f"Collection '{self.collection_name}' does not exist")

                collection_info = await self.database.get_collection_info(self.collection_name)
                if collection_info["count"] == 0:
                    raise TestError("Database is empty. No documents to query.")

            # Generate query embedding
            with trace_span("tester.embedding_query", attributes={"query": query[:50]}):
                logger.debug("Generating embedding for query: %s", query[:50])
                embeddings = await self.embedding_provider.embed_texts([query])
                query_embedding = embeddings[0]

            # Validate dimension match
            expected_dim = collection_info["dimension"]
            actual_dim = len(query_embedding)
            if actual_dim != expected_dim:
                raise TestError(f"Dimension mismatch: model={actual_dim}, database={expected_dim}")

            # Query database
            with trace_span(
                "tester.database_search",
                attributes={
                    "collection": self.collection_name,
                    "top_k": top_k,
                },
            ):
                logger.debug("Querying database with top_k=%d", top_k)
                raw_results = await self.database.query(
                    collection=self.collection_name,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )

            # Format results
            results = []
            for rank, result in enumerate(raw_results, start=1):
                results.append(
                    {
                        "rank": rank,
                        "id": result["id"],
                        "text": result["text"],
                        "score": result["score"],
                    }
                )

            # Check if we got fewer results than requested
            if len(results) < top_k and len(results) == collection_info["count"]:
                logger.info(
                    "Requested top-k (%d) exceeds collection size (%d), returning all documents",
                    top_k,
                    collection_info["count"],
                )

            elapsed_time = time.time() - start_time

            # Log query execution
            logger.info("Query: %s", query)
            logger.info("Results: %d", len(results))
            logger.debug("Top-K: %d", top_k)

            # Trace query completion
            with trace_span(
                "tester.query_complete",
                attributes={
                    "query": query[:50],
                    "top_k": top_k,
                    "duration": elapsed_time,
                    "result_count": len(results),
                },
            ):
                pass

            return {
                "query": query,
                "results": results,
                "tokens": 0,  # Local models don't consume tokens
                "time": elapsed_time,
            }

        except DatabaseError as e:
            logger.error("Database error during query: %s", e)
            raise TestError(f"Database operation failed: {e}") from e

        except EmbeddingError as e:
            logger.error("Embedding error during query: %s", e)
            raise TestError(f"Embedding generation failed: {e}") from e

    def format_results(self, result_data: dict[str, Any], output_format: str) -> str:
        """Format query results in the specified output format.

        Args:
            result_data: Result dictionary from test_query
            output_format: Output format (table, json, text)

        Returns:
            Formatted string ready for display

        Raises:
            ValidationError: If format is invalid
        """
        self._validate_format(output_format)

        if output_format == "table":
            return self._format_table(result_data)
        elif output_format == "json":
            return self._format_json(result_data)
        elif output_format == "text":
            return self._format_text(result_data)
        else:
            raise ValidationError(f"Invalid format: {output_format}")

    def _format_table(self, result_data: dict[str, Any]) -> str:
        """Format results as a rich table.

        Args:
            result_data: Result dictionary from test_query

        Returns:
            Rich table as string
        """
        console = Console()
        table = Table(title=f"Query: {result_data['query']}")

        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("ID", style="magenta")
        table.add_column("Text", style="white")
        table.add_column("Score", justify="right", style="green")

        for result in result_data["results"]:
            # Truncate text to 80 characters
            text = result["text"]
            if len(text) > 80:
                text = text[:77] + "..."

            table.add_row(
                str(result["rank"]),
                result["id"],
                text,
                f"{result['score']:.4f}",
            )

        # Capture table output
        with console.capture() as capture:
            console.print(table)
            console.print()
            console.print(f"Tokens: {result_data['tokens']}")
            console.print(f"Time: {result_data['time']:.2f}s")

        return capture.get()

    def _format_json(self, result_data: dict[str, Any]) -> str:
        """Format results as JSON.

        Args:
            result_data: Result dictionary from test_query

        Returns:
            JSON string
        """
        return json.dumps(result_data, indent=2)

    def _format_text(self, result_data: dict[str, Any]) -> str:
        """Format results as plain text.

        Args:
            result_data: Result dictionary from test_query

        Returns:
            Plain text string
        """
        lines = [f"Query: {result_data['query']}", ""]

        for result in result_data["results"]:
            lines.append(f"{result['rank']}. [{result['id']}] (score: {result['score']:.2f})")
            lines.append(result["text"])
            lines.append("")

        lines.append(f"Tokens: {result_data['tokens']}")
        lines.append(f"Time: {result_data['time']:.2f}s")

        return "\n".join(lines)

    def _validate_query(self, query: str) -> None:
        """Validate query string.

        Args:
            query: Query text to validate

        Raises:
            ValidationError: If query is invalid
        """
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

    def _validate_top_k(self, top_k: int) -> None:
        """Validate top-k parameter.

        Args:
            top_k: Top-k value to validate

        Raises:
            ValidationError: If top-k is invalid
        """
        if not isinstance(top_k, int):
            raise ValidationError(f"Top-K must be an integer, got {type(top_k).__name__}")

        if top_k < 1 or top_k > 100:
            raise ValidationError("Top-K must be between 1 and 100")

    def _validate_format(self, output_format: str) -> None:
        """Validate output format.

        Args:
            output_format: Format to validate

        Raises:
            ValidationError: If format is invalid
        """
        valid_formats = {"table", "json", "text"}
        if output_format not in valid_formats:
            raise ValidationError(f"Invalid format. Must be one of: {', '.join(sorted(valid_formats))}")
