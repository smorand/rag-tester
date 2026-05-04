"""Unit tests for the answer command."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from rag_tester.providers.llm.base import LLMResponse
from rag_tester.rag_tester import app

runner = CliRunner()


@pytest.fixture
def mock_providers(mocker: Any) -> dict[str, Any]:
    """Mock all three providers used by the answer command."""
    embedding = mocker.patch("rag_tester.commands.answer.LocalEmbeddingProvider")
    embedding_instance = embedding.return_value
    embedding_instance.embed_texts = AsyncMock(return_value=[[0.1] * 384])

    db_factory = mocker.patch("rag_tester.commands.answer.get_database_provider")
    db_instance = db_factory.return_value
    db_instance.query = AsyncMock(
        return_value=[
            {"id": "doc1", "text": "RAG combines retrieval and generation.", "score": 0.91},
            {"id": "doc2", "text": "Embeddings live in vector databases.", "score": 0.83},
        ]
    )

    llm_factory = mocker.patch("rag_tester.commands.answer.get_llm_provider")
    llm_instance = llm_factory.return_value
    llm_instance.complete = AsyncMock(
        return_value=LLMResponse(text="RAG retrieves context then generates an answer [1].", tokens=120)
    )
    llm_instance.get_model_name.return_value = "openai/gpt-4o-mini"

    return {
        "embedding": embedding,
        "db_factory": db_factory,
        "db": db_instance,
        "llm_factory": llm_factory,
        "llm": llm_instance,
    }


class TestAnswerCommand:
    def test_success_default_top_k(self, mock_providers: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "answer",
                "What is RAG?",
                "--database",
                "chromadb://localhost:8000/coll",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )
        assert result.exit_code == 0
        assert "RAG retrieves context" in result.stdout
        # Default top-k=5 must be passed to the database
        assert mock_providers["db"].query.await_count == 1
        assert mock_providers["db"].query.await_args.kwargs["top_k"] == 5
        # The LLM must have been called with both prompts
        assert mock_providers["llm"].complete.await_count == 1
        kwargs = mock_providers["llm"].complete.await_args.kwargs
        assert "system_prompt" in kwargs
        assert "user_prompt" in kwargs
        assert "What is RAG?" in kwargs["user_prompt"]

    def test_custom_top_k_propagates(self, mock_providers: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "answer",
                "Q",
                "-d",
                "chromadb://localhost:8000/coll",
                "-e",
                "model",
                "--top-k",
                "12",
            ],
        )
        assert result.exit_code == 0
        assert mock_providers["db"].query.await_args.kwargs["top_k"] == 12

    def test_custom_llm_model_overrides_settings(self, mock_providers: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "answer",
                "Q",
                "-d",
                "chromadb://localhost:8000/coll",
                "-e",
                "model",
                "--llm-model",
                "anthropic/claude-sonnet-4-5",
            ],
        )
        assert result.exit_code == 0
        # Factory must be called with the explicit model
        kwargs = mock_providers["llm_factory"].call_args.kwargs
        assert kwargs["model_name"] == "anthropic/claude-sonnet-4-5"

    def test_empty_query_rejected(self, mock_providers: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "answer",
                "   ",
                "-d",
                "chromadb://localhost:8000/coll",
                "-e",
                "model",
            ],
        )
        assert result.exit_code == 1
        assert "query cannot be empty" in (result.stderr + result.stdout)

    @pytest.mark.parametrize("top_k", [0, -1, 200])
    def test_top_k_out_of_range_rejected(self, mock_providers: dict[str, Any], top_k: int) -> None:
        result = runner.invoke(
            app,
            [
                "answer",
                "Q",
                "-d",
                "chromadb://localhost:8000/coll",
                "-e",
                "model",
                "--top-k",
                str(top_k),
            ],
        )
        assert result.exit_code == 1
        assert "top-k" in (result.stderr + result.stdout)

    def test_unknown_llm_provider_rejected(self, mock_providers: dict[str, Any]) -> None:
        mock_providers["llm_factory"].side_effect = ValueError("Unsupported LLM provider 'voyage-ai'")
        result = runner.invoke(
            app,
            [
                "answer",
                "Q",
                "-d",
                "chromadb://localhost:8000/coll",
                "-e",
                "model",
                "--llm",
                "voyage-ai",
            ],
        )
        assert result.exit_code == 1
        assert "Unsupported LLM provider" in (result.stderr + result.stdout)

    def test_renders_sources(self, mock_providers: dict[str, Any]) -> None:
        result = runner.invoke(
            app,
            [
                "answer",
                "What is RAG?",
                "-d",
                "chromadb://localhost:8000/coll",
                "-e",
                "model",
            ],
        )
        assert result.exit_code == 0
        # Both source IDs must appear in the rendered output
        assert "doc1" in result.stdout
        assert "doc2" in result.stdout
