"""Answer command: full RAG round-trip (retrieval + generation).

Embed the query, fetch the top-k most similar records from the vector
database, build a prompt that injects those records as numbered context, send
it to an LLM, and render the final answer with citations.

The retrieval-only equivalent is :func:`rag_tester.commands.test.test_command`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel

from rag_tester.config import get_settings
from rag_tester.providers.databases import get_database_provider
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider
from rag_tester.providers.llm import LLMError, get_llm_provider
from rag_tester.tracing import trace_span

if TYPE_CHECKING:
    from rag_tester.providers.databases.base import VectorDatabase

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions strictly from the provided context. "
    "If the context does not contain the answer, say so explicitly. "
    "Cite sources by their numeric id in square brackets, e.g. [2]."
)


def _build_user_prompt(query: str, sources: list[dict[str, Any]]) -> str:
    """Compose the user message: numbered context block + the query."""
    if not sources:
        context_block = "(no context retrieved)"
    else:
        context_block = "\n\n".join(f"[{i + 1}] (id={src['id']}) {src['text']}" for i, src in enumerate(sources))
    return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"


def answer_command(
    query: str = typer.Argument(..., help="Natural-language question to answer"),
    database: str = typer.Option(
        ...,
        "--database",
        "-d",
        help="Database connection string (e.g., chromadb://localhost:8000/collection)",
    ),
    embedding: str = typer.Option(
        ...,
        "--embedding",
        "-e",
        help="Embedding model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of context chunks to retrieve (1-100)"),
    llm: str = typer.Option("openrouter", "--llm", "-l", help="LLM provider id (currently: openrouter)"),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        "-m",
        help="LLM model id (e.g., openai/gpt-4o-mini); defaults to RAG_TESTER_LLM_MODEL",
    ),
) -> None:
    """Answer a question using retrieved context (RAG).

    Examples:
        # Default model (openai/gpt-4o-mini via OpenRouter)
        rag-tester answer "What is RAG?" -d chromadb://localhost:8000/coll -e sentence-transformers/all-MiniLM-L6-v2

        # Pick a specific model and increase context size
        rag-tester answer "Explain pgvector" -d postgresql://u:p@h:5432/db/tbl -e sentence-transformers/all-MiniLM-L6-v2 --llm-model anthropic/claude-sonnet-4-5 --top-k 10
    """
    try:
        asyncio.get_running_loop()
        import nest_asyncio

        nest_asyncio.apply()
    except RuntimeError:
        pass

    exit_code = asyncio.run(
        _answer_async(
            query=query,
            database=database,
            embedding=embedding,
            top_k=top_k,
            llm=llm,
            llm_model=llm_model,
        )
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


async def _answer_async(
    query: str,
    database: str,
    embedding: str,
    top_k: int,
    llm: str,
    llm_model: str | None,
) -> int:
    """Async pipeline behind ``answer_command``. Returns the CLI exit code."""
    if top_k < 1 or top_k > 100:
        error_console.print("[red]Error: --top-k must be between 1 and 100[/red]")
        return 1

    if not query.strip():
        error_console.print("[red]Error: query cannot be empty[/red]")
        return 1

    # Resolve the LLM model (CLI flag wins over Settings default)
    resolved_llm_model = llm_model or get_settings().llm_model

    # 1. Initialize providers
    try:
        embedding_provider = LocalEmbeddingProvider(model_name=embedding)
    except Exception as e:
        error_console.print(f"[red]Error: Failed to load embedding model: {embedding}[/red]")
        logger.error("Failed to load embedding model: %s", e)
        return 1

    try:
        db_provider: VectorDatabase = get_database_provider(database)
    except ValueError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        return 1
    except Exception as e:
        error_console.print(f"[red]Error: Database connection failed: {e}[/red]")
        logger.error("Database connection failed: %s", e)
        return 1

    try:
        llm_provider = get_llm_provider(name=llm, model_name=resolved_llm_model)
    except ValueError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        return 1
    except LLMError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        return 1

    collection_name = database.rsplit("/", 1)[-1]

    # 2. Retrieve top-k context
    start_time = time.time()
    with trace_span(
        "answer.retrieve",
        attributes={"collection": collection_name, "top_k": top_k, "query.length": len(query)},
    ):
        try:
            embeddings = await embedding_provider.embed_texts([query])
            query_embedding = embeddings[0]
            sources = await db_provider.query(
                collection=collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as e:
            error_console.print(f"[red]Error: Retrieval failed: {e}[/red]")
            logger.exception("Retrieval failed")
            return 1

    retrieval_time = time.time() - start_time

    # 3. Build prompt and call LLM
    user_prompt = _build_user_prompt(query, sources)
    with trace_span(
        "answer.generate",
        attributes={"llm.provider": llm, "llm.model": resolved_llm_model, "context.sources": len(sources)},
    ):
        try:
            llm_response = await llm_provider.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except LLMError as e:
            error_console.print(f"[red]Error: LLM call failed: {e}[/red]")
            logger.exception("LLM call failed")
            return 1

    total_time = time.time() - start_time

    # 4. Render
    console.print(Panel(llm_response.text.strip(), title="Answer", border_style="green"))

    if sources:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for i, src in enumerate(sources, start=1):
            score_str = f"score={src['score']:.4f}" if "score" in src else ""
            console.print(f"  [{i}] {src['id']} {score_str}")

    console.print()
    console.print(
        f"[dim]Retrieval: {retrieval_time:.2f}s   "
        f"Total: {total_time:.2f}s   "
        f"Tokens: {llm_response.tokens}   "
        f"Model: {resolved_llm_model}[/dim]"
    )
    return 0
