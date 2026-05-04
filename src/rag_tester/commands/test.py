"""Test command implementation for RAG Tester.

This module provides the CLI command for testing queries against a vector database.
"""

import asyncio
import logging

import typer
from rich.console import Console

from rag_tester.core.tester import Tester, TestError, ValidationError
from rag_tester.providers.databases.base import DatabaseError, VectorDatabase
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
    # Check if we're already in an event loop (e.g., called from async tests)
    try:
        asyncio.get_running_loop()
        # We're in an event loop, use nest_asyncio to allow nested event loops
        import nest_asyncio

        nest_asyncio.apply()
        exit_code = asyncio.run(
            _test_async(
                query=query,
                database=database,
                embedding=embedding,
                top_k=top_k,
                output_format=format,
            )
        )
    except RuntimeError:
        # No event loop running, use asyncio.run()
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
        # Parse database connection string and determine provider
        # Supported formats:
        # - chromadb://host:port/collection_name
        # - postgresql://user:pass@host:port/dbname/table_name
        # - sqlite:///path/to/db.db/table_name

        if database.startswith("chromadb://"):
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

            _host, port_str = host_port.rsplit(":", 1)
            try:
                int(port_str)
            except ValueError:
                error_console.print(f"[red]Error: Invalid port number: {port_str}[/red]")
                return 1

        elif database.startswith("postgresql://"):
            # Extract table name (last part after /)
            remainder = database.replace("postgresql://", "")
            parts = remainder.rsplit("/", 1)
            if len(parts) != 2:
                error_console.print(
                    "[red]Error: Invalid PostgreSQL connection string. Expected format: postgresql://user:pass@host:port/dbname/table_name[/red]"
                )
                return 1
            collection_name = parts[1]

        elif database.startswith("sqlite://"):
            # Extract table name (last part after /)
            remainder = database.replace("sqlite://", "")
            parts = remainder.rsplit("/", 1)
            if len(parts) != 2:
                error_console.print(
                    "[red]Error: Invalid SQLite connection string. Expected format: sqlite:///path/to/db.db/table_name[/red]"
                )
                return 1
            collection_name = parts[1]

        elif database.startswith("milvus://"):
            # Extract collection name (last part after /)
            remainder = database.replace("milvus://", "")
            parts = remainder.rsplit("/", 1)
            if len(parts) != 2:
                error_console.print(
                    "[red]Error: Invalid Milvus connection string. Expected format: milvus://host:port/collection_name[/red]"
                )
                return 1
            collection_name = parts[1]

        else:
            error_console.print(
                "[red]Error: Unsupported database. Use chromadb://..., postgresql://..., sqlite://..., or milvus://...[/red]"
            )
            return 1

        # Initialize providers
        logger.info("Initializing embedding provider: %s", embedding)

        try:
            embedding_provider = LocalEmbeddingProvider(model_name=embedding)
        except Exception as e:
            error_console.print(f"[red]Error: Failed to load embedding model: {embedding}[/red]")
            logger.error("Failed to load embedding model: %s", e)
            return 1

        logger.info("Connecting to database: %s", database)

        try:
            # Instantiate the appropriate database provider
            db_provider: VectorDatabase
            if database.startswith("chromadb://"):
                db_provider = ChromaDBProvider(connection_string=database)
            elif database.startswith("postgresql://"):
                from rag_tester.providers.databases.postgresql import PostgreSQLProvider

                db_provider = PostgreSQLProvider(connection_string=database)
            elif database.startswith("sqlite://"):
                from rag_tester.providers.databases.sqlite import SQLiteProvider

                db_provider = SQLiteProvider(connection_string=database)
            elif database.startswith("milvus://"):
                from rag_tester.providers.databases.milvus import MilvusProvider

                db_provider = MilvusProvider(connection_string=database)
            elif database.startswith("elasticsearch://"):
                from rag_tester.providers.databases.elasticsearch import ElasticsearchProvider

                db_provider = ElasticsearchProvider(connection_string=database)
            else:
                error_console.print("[red]Error: Unsupported database provider[/red]")
                return 1
        except Exception as e:
            error_console.print(f"[red]Error: Database connection failed: {e}[/red]")
            logger.error("Database connection failed: %s", e)
            return 1

        # Create tester
        tester = Tester(
            database=db_provider,
            embedding_provider=embedding_provider,
            collection_name=collection_name,
        )

        # Execute query
        logger.info("Executing query: %s", query)
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
        logger.error("Validation error: %s", e)
        return 1

    except TestError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error("Test error: %s", e)
        return 1

    except DatabaseError as e:
        error_console.print(f"[red]Error: Database operation failed: {e}[/red]")
        logger.error("Database error: %s", e)
        return 1

    except EmbeddingError as e:
        error_console.print(f"[red]Error: Embedding generation failed: {e}[/red]")
        logger.error("Embedding error: %s", e)
        return 1

    except Exception as e:
        error_console.print(f"[red]Error: Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during test")
        return 1
