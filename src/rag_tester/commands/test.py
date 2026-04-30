"""Test command implementation for RAG Tester.

This module provides the CLI command for testing queries against a vector database.
"""

import asyncio
import logging

import typer
from rich.console import Console

from rag_tester.core.tester import TestError, Tester, ValidationError
from rag_tester.providers.databases.base import DatabaseError
from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.embeddings.base import EmbeddingError
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


def test_command(
    query: str = typer.Argument(..., help="Query text to test"),
    database: str = typer.Option(
        ..., "--database", "-d", help="Database connection string (e.g., chromadb://localhost:8000/collection)"
    ),
    embedding: str = typer.Option(
        ..., "--embedding", "-e", help="Embedding model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return (1-100)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, text"),
) -> None:
    """Test a query against the vector database.

    This command executes a single query and displays results in the specified format.
    Use this to interactively test your RAG system and validate retrieval quality.

    Examples:
        # Test with table output (default)
        rag-tester test "What is machine learning?" --database chromadb://localhost:8000/my_collection --embedding sentence-transformers/all-MiniLM-L6-v2

        # Test with JSON output
        rag-tester test "machine learning" -d chromadb://localhost:8000/my_collection -e sentence-transformers/all-MiniLM-L6-v2 --format json

        # Test with custom top-k
        rag-tester test "AI concepts" -d chromadb://localhost:8000/my_collection -e sentence-transformers/all-MiniLM-L6-v2 --top-k 10
    """
    # Run async test function
    exit_code = asyncio.run(
        _test_async(
            query=query,
            database=database,
            embedding=embedding,
            top_k=top_k,
            output_format=format,
        )
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


async def _test_async(
    query: str,
    database: str,
    embedding: str,
    top_k: int,
    output_format: str,
) -> int:
    """Async implementation of test command.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse database connection string
        # Format: chromadb://host:port/collection_name
        if not database.startswith("chromadb://"):
            error_console.print(
                "[red]Error: Only ChromaDB is currently supported. Use chromadb://host:port/collection[/red]"
            )
            return 1

        db_parts = database.replace("chromadb://", "").split("/")
        if len(db_parts) != 2:
            error_console.print(
                "[red]Error: Invalid database connection string. Expected format: chromadb://host:port/collection[/red]"
            )
            return 1

        host_port = db_parts[0]
        collection_name = db_parts[1]

        if ":" not in host_port:
            error_console.print(
                "[red]Error: Invalid database connection string. Expected format: chromadb://host:port/collection[/red]"
            )
            return 1

        host, port_str = host_port.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            error_console.print(f"[red]Error: Invalid port number: {port_str}[/red]")
            return 1

        # Initialize providers
        logger.info(f"Initializing embedding provider: {embedding}")

        try:
            embedding_provider = LocalEmbeddingProvider(model_name=embedding)
        except Exception as e:
            error_console.print(f"[red]Error: Failed to load embedding model: {embedding}[/red]")
            logger.error(f"Failed to load embedding model: {e}")
            return 1

        logger.info(f"Connecting to database: {host}:{port}")

        try:
            db_provider = ChromaDBProvider(host=host, port=port)
        except Exception as e:
            error_console.print(f"[red]Error: Database connection failed: {e}[/red]")
            logger.error(f"Database connection failed: {e}")
            return 1

        # Create tester
        tester = Tester(
            database=db_provider,
            embedding_provider=embedding_provider,
            collection_name=collection_name,
        )

        # Execute query
        logger.info(f"Executing query: {query}")
        result_data = await tester.test_query(
            query=query,
            top_k=top_k,
            output_format=output_format,
        )

        # Format and display results
        formatted_output = tester.format_results(result_data, output_format)
        console.print(formatted_output)

        return 0

    except ValidationError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Validation error: {e}")
        return 1

    except TestError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Test error: {e}")
        return 1

    except DatabaseError as e:
        error_console.print(f"[red]Error: Database operation failed: {e}[/red]")
        logger.error(f"Database error: {e}")
        return 1

    except EmbeddingError as e:
        error_console.print(f"[red]Error: Embedding generation failed: {e}[/red]")
        logger.error(f"Embedding error: {e}")
        return 1

    except Exception as e:
        error_console.print(f"[red]Error: Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during test")
        return 1
