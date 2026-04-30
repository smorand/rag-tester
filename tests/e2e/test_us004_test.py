"""E2E acceptance tests for US-004: Test Command.

These tests validate the complete test command workflow including:
- Query execution with multiple output formats
- Input validation and error handling
- Performance requirements
- Edge cases
"""

import json
import time

import pytest

from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider


@pytest.fixture
async def test_collection_with_data(chromadb_server):
    """Create a test collection with 100 documents."""
    host, port = chromadb_server
    db = ChromaDBProvider(host=host, port=port)
    embedding_provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    collection_name = "test_collection"
    dimension = embedding_provider.get_dimension()

    # Create collection
    await db.create_collection(collection_name, dimension)

    # Generate 100 test documents
    records = []
    texts = []
    for i in range(100):
        if i == 42:
            # Special document for testing
            text = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        else:
            text = f"This is test document number {i} with some content about various topics."

        records.append(
            {
                "id": f"doc{i}",
                "text": text,
                "metadata": {"index": i},
            }
        )
        texts.append(text)

    # Generate embeddings
    embeddings = await embedding_provider.embed_texts(texts)

    # Add embeddings to records
    for record, embedding in zip(records, embeddings, strict=True):
        record["embedding"] = embedding

    # Insert records
    await db.insert(collection_name, records)

    yield collection_name, host, port

    # Cleanup
    await db.delete_collection(collection_name)


@pytest.fixture
async def small_collection(chromadb_server):
    """Create a small collection with 50 documents."""
    host, port = chromadb_server
    db = ChromaDBProvider(host=host, port=port)
    embedding_provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    collection_name = "small_collection"
    dimension = embedding_provider.get_dimension()

    await db.create_collection(collection_name, dimension)

    records = []
    texts = [f"Document {i}" for i in range(50)]
    embeddings = await embedding_provider.embed_texts(texts)

    for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=True)):
        records.append(
            {
                "id": f"doc{i}",
                "text": text,
                "embedding": embedding,
                "metadata": {},
            }
        )

    await db.insert(collection_name, records)

    yield collection_name, host, port

    await db.delete_collection(collection_name)


@pytest.fixture
async def empty_collection(chromadb_server):
    """Create an empty collection."""
    host, port = chromadb_server
    db = ChromaDBProvider(host=host, port=port)
    embedding_provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    collection_name = "empty_collection"
    dimension = embedding_provider.get_dimension()

    await db.create_collection(collection_name, dimension)

    yield collection_name, host, port

    await db.delete_collection(collection_name)


@pytest.fixture
async def exact_match_collection(chromadb_server):
    """Create a collection with a known document for exact matching."""
    host, port = chromadb_server
    db = ChromaDBProvider(host=host, port=port)
    embedding_provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    collection_name = "exact_match"
    dimension = embedding_provider.get_dimension()

    await db.create_collection(collection_name, dimension)

    text = "Python is a programming language"
    embeddings = await embedding_provider.embed_texts([text])

    records = [
        {
            "id": "doc1",
            "text": text,
            "embedding": embeddings[0],
            "metadata": {},
        }
    ]

    await db.insert(collection_name, records)

    yield collection_name, host, port, text

    await db.delete_collection(collection_name)


class TestHappyPath:
    """Happy path acceptance tests."""

    @pytest.mark.asyncio
    async def test_e2e_002_manual_query_test(self, cli_runner, test_collection_with_data):
        """E2E-002: Manual Query Test - Critical.

        Test successful query execution with table output format.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "What is machine learning?",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "3",
                "--format",
                "table",
            ]
        )

        # Verify exit code
        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Verify output contains table elements
        assert "Rank" in result.stdout
        assert "ID" in result.stdout
        assert "Text" in result.stdout
        assert "Score" in result.stdout

        # Verify doc42 is in results (should be top result for ML query)
        assert "doc42" in result.stdout

        # Verify results are sorted by score (check that scores decrease)
        # This is implicit in the table format

        # Verify metadata
        assert "Tokens:" in result.stdout
        assert "Time:" in result.stdout

    @pytest.mark.asyncio
    async def test_e2e_018_json_output_format(self, cli_runner, test_collection_with_data):
        """E2E-018: JSON Output Format - Medium.

        Test query with JSON output format.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "machine learning",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--format",
                "json",
            ]
        )

        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Parse JSON output
        output = json.loads(result.stdout)

        # Verify structure
        assert "query" in output
        assert output["query"] == "machine learning"
        assert "results" in output
        assert "tokens" in output
        assert "time" in output

        # Verify results array
        assert len(output["results"]) == 5  # Default top-k

        # Verify each result has required fields
        for result_item in output["results"]:
            assert "rank" in result_item
            assert "id" in result_item
            assert "text" in result_item
            assert "score" in result_item

    @pytest.mark.asyncio
    async def test_e2e_019_text_output_format(self, cli_runner, test_collection_with_data):
        """E2E-019: Text Output Format - Medium.

        Test query with plain text output format.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "machine learning",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--format",
                "text",
            ]
        )

        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Verify text format structure
        assert "Query: machine learning" in result.stdout

        # Verify numbered results (1-5)
        for i in range(1, 6):
            assert f"{i}. [" in result.stdout
            assert "(score:" in result.stdout

        # Verify metadata
        assert "Tokens:" in result.stdout
        assert "Time:" in result.stdout

    @pytest.mark.asyncio
    async def test_e2e_020_custom_top_k(self, cli_runner, test_collection_with_data):
        """E2E-020: Custom Top-K - Medium.

        Test query with custom top-k value.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "machine learning",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "10",
                "--format",
                "json",
            ]
        )

        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Parse JSON output
        output = json.loads(result.stdout)

        # Verify exactly 10 results
        assert len(output["results"]) == 10

        # Verify results are sorted by score (descending)
        scores = [r["score"] for r in output["results"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_e2e_066_query_latency(self, cli_runner, test_collection_with_data):
        """E2E-066: Query Latency - Medium (performance baseline).

        Test that query completes within acceptable time.
        """
        collection_name, host, port = test_collection_with_data

        start_time = time.time()

        result = cli_runner.invoke(
            [
                "test",
                "machine learning",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--format",
                "json",
            ]
        )

        elapsed_time = time.time() - start_time

        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Verify total time < 1 second (local model)
        assert elapsed_time < 1.0, f"Query took {elapsed_time:.2f}s, expected < 1.0s"

        # Verify time is displayed in output
        output = json.loads(result.stdout)
        assert "time" in output
        assert output["time"] > 0


class TestEdgeCases:
    """Edge case acceptance tests."""

    @pytest.mark.asyncio
    async def test_e2e_085_top_k_exceeds_collection_size(self, cli_runner, small_collection):
        """E2E-085: Top-K Exceeds Collection Size - Medium.

        Test query when top-k exceeds number of documents.
        """
        collection_name, host, port = small_collection

        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "100",
                "--format",
                "json",
            ]
        )

        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Parse output
        output = json.loads(result.stdout)

        # Should return exactly 50 results (all documents)
        assert len(output["results"]) == 50

    @pytest.mark.asyncio
    async def test_e2e_087_perfect_score(self, cli_runner, exact_match_collection):
        """E2E-087: Perfect Score (1.0) - Low.

        Test exact match query returns score ≈ 1.0.
        """
        collection_name, host, port, text = exact_match_collection

        result = cli_runner.invoke(
            [
                "test",
                text,  # Query with exact same text
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--format",
                "json",
            ]
        )

        assert result.exit_code == 0, f"Command failed: {result.stderr}"

        # Parse output
        output = json.loads(result.stdout)

        # Verify top result has score ≈ 1.0 (within 0.01)
        assert len(output["results"]) > 0
        top_result = output["results"][0]
        assert top_result["id"] == "doc1"
        assert abs(top_result["score"] - 1.0) < 0.01, f"Expected score ≈ 1.0, got {top_result['score']}"


class TestErrorCases:
    """Error case acceptance tests."""

    @pytest.mark.asyncio
    async def test_e2e_test_001_empty_database(self, cli_runner, empty_collection):
        """E2E-TEST-001: Empty Database - Critical.

        Test query against empty collection.
        """
        collection_name, host, port = empty_collection

        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
        )

        assert result.exit_code == 1
        assert "Database is empty" in result.stderr

    @pytest.mark.asyncio
    async def test_e2e_test_002_database_unreachable(self, cli_runner):
        """E2E-TEST-002: Database Unreachable - Critical.

        Test query when database is unreachable.
        """
        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:9999/test",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
        )

        assert result.exit_code == 1
        assert "Database connection failed" in result.stderr

    @pytest.mark.asyncio
    async def test_e2e_test_003_embedding_api_failure(self, cli_runner, test_collection_with_data):
        """E2E-TEST-003: Embedding API Failure - Critical.

        Test query with invalid embedding model.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "invalid/model",
            ]
        )

        assert result.exit_code == 1
        assert "Failed to load embedding model" in result.stderr

    @pytest.mark.asyncio
    async def test_e2e_test_004_dimension_mismatch(self, cli_runner, chromadb_server):
        """E2E-TEST-004: Dimension Mismatch - High.

        Test query when model dimension doesn't match collection.
        """
        host, port = chromadb_server
        db = ChromaDBProvider(host=host, port=port)

        # Create collection with dimension 768
        collection_name = "dim768_collection"
        await db.create_collection(collection_name, dimension=768)

        # Add a dummy record with 768-dim embedding
        records = [
            {
                "id": "doc1",
                "text": "Test",
                "embedding": [0.1] * 768,
                "metadata": {},
            }
        ]
        await db.insert(collection_name, records)

        try:
            # Try to query with 384-dim model
            result = cli_runner.invoke(
                [
                    "test",
                    "test query",
                    "--database",
                    f"chromadb://{host}:{port}/{collection_name}",
                    "--embedding",
                    "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim
                ]
            )

            assert result.exit_code == 1
            assert "Dimension mismatch" in result.stderr
            assert "model=384" in result.stderr
            assert "database=768" in result.stderr
        finally:
            await db.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_e2e_test_005_invalid_top_k_zero(self, cli_runner, test_collection_with_data):
        """E2E-TEST-005: Invalid Top-K (Zero) - High.

        Test query with top-k = 0.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "0",
            ]
        )

        assert result.exit_code == 1
        assert "Top-K must be between 1 and 100" in result.stderr

    @pytest.mark.asyncio
    async def test_e2e_test_006_invalid_top_k_too_large(self, cli_runner, test_collection_with_data):
        """E2E-TEST-006: Invalid Top-K (Too Large) - High.

        Test query with top-k > 100.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "101",
            ]
        )

        assert result.exit_code == 1
        assert "Top-K must be between 1 and 100" in result.stderr

    @pytest.mark.asyncio
    async def test_e2e_test_007_invalid_output_format(self, cli_runner, test_collection_with_data):
        """E2E-TEST-007: Invalid Output Format - High.

        Test query with invalid output format.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "test query",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--format",
                "xml",
            ]
        )

        assert result.exit_code == 1
        assert "Invalid format" in result.stderr

    @pytest.mark.asyncio
    async def test_e2e_test_008_empty_query_string(self, cli_runner, test_collection_with_data):
        """E2E-TEST-008: Empty Query String - High.

        Test query with empty string.
        """
        collection_name, host, port = test_collection_with_data

        result = cli_runner.invoke(
            [
                "test",
                "",
                "--database",
                f"chromadb://{host}:{port}/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
        )

        assert result.exit_code == 1
        assert "Query cannot be empty" in result.stderr
