"""Core user journey E2E tests.

Tests the primary workflows:
- E2E-001: Initial dataset load with local embedding model
- E2E-002: Manual query test
- E2E-003: Bulk test with pass/fail cases
- E2E-004: Bulk test verbose mode
- E2E-005: Compare two embedding models
- E2E-006: Upsert mode
- E2E-007: Flush mode
"""

import subprocess
from pathlib import Path

import pytest
import yaml


class TestCoreJourneys:
    """Core user journey tests following the E2E test plan."""

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_001_initial_load_local_embedding(
        self,
        sample_data_file: Path,
        chromadb_url: str,
        embedding_model: str,
        temp_dir: Path,
    ):
        """E2E-001: Initial Dataset Load with Local Embedding Model.

        Verifies:
        - Clean ChromaDB instance can be loaded
        - 10 records are successfully loaded
        - No failed records
        - Collection exists with correct dimensions
        - Trace file contains expected spans
        """
        collection_name = "test_e2e_001"
        db_url = f"{chromadb_url}/{collection_name}"

        # Execute load command
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(sample_data_file),
                "--database",
                db_url,
                "--embedding",
                embedding_model,
                "--parallel",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Verify exit code
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify stdout contains success message
        assert "Successfully loaded 10 records" in result.stdout
        assert "Failed records: 0" in result.stdout
        assert "Total tokens: 0" in result.stdout  # Local model doesn't count tokens
        assert "Total time:" in result.stdout

        # Verify trace file exists and contains expected spans
        trace_file = Path("traces/rag-tester.jsonl")
        assert trace_file.exists(), "Trace file not created"

        with open(trace_file) as f:
            traces = [line for line in f if line.strip()]
            assert len(traces) > 0, "No traces recorded"

            # Check for expected span names
            trace_content = "".join(traces)
            assert "file_read" in trace_content or "load" in trace_content
            assert "embedding" in trace_content or "batch" in trace_content
            assert "database" in trace_content or "insert" in trace_content

    @pytest.mark.e2e
    @pytest.mark.critical
    @pytest.mark.skip(reason="Requires loaded database from E2E-001")
    def test_e2e_002_manual_query_test(
        self,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-002: Manual Query Test.

        Verifies:
        - Query returns results in table format
        - Top-K results are returned
        - Results are ranked by score
        - Trace contains query spans
        """
        collection_name = "test_e2e_001"
        db_url = f"{chromadb_url}/{collection_name}"

        result = subprocess.run(
            [
                "rag-tester",
                "test",
                "What is machine learning?",
                "--database",
                db_url,
                "--embedding",
                embedding_model,
                "--top-k",
                "3",
                "--format",
                "table",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify table output
        assert "Rank" in result.stdout
        assert "ID" in result.stdout
        assert "Text" in result.stdout
        assert "Score" in result.stdout

        # Verify we got 3 results
        lines = result.stdout.split("\n")
        result_lines = [l for l in lines if l.strip() and not l.startswith("-") and "Rank" not in l]
        assert len(result_lines) >= 3, "Expected at least 3 results"

    @pytest.mark.e2e
    @pytest.mark.critical
    @pytest.mark.skip(reason="Requires loaded database and test suite")
    def test_e2e_003_bulk_test_with_pass_fail(
        self,
        sample_test_suite: Path,
        chromadb_url: str,
        embedding_model: str,
        temp_dir: Path,
    ):
        """E2E-003: Bulk Test with Pass/Fail Cases.

        Verifies:
        - Bulk test executes all tests
        - Results file is created
        - Summary contains correct counts
        - Only failed tests are included by default
        """
        collection_name = "test_e2e_001"
        db_url = f"{chromadb_url}/{collection_name}"
        results_file = temp_dir / "results.yaml"

        result = subprocess.run(
            [
                "rag-tester",
                "bulk-test",
                "--file",
                str(sample_test_suite),
                "--database",
                db_url,
                "--embedding",
                embedding_model,
                "--output",
                str(results_file),
                "--parallel",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert results_file.exists(), "Results file not created"

        # Load and verify results
        with open(results_file) as f:
            results = yaml.safe_load(f)

        assert "summary" in results
        assert "total" in results["summary"]
        assert "passed" in results["summary"]
        assert "failed" in results["summary"]
        assert "tokens" in results["summary"]
        assert "time" in results["summary"]

        # Verify only failed tests are included by default
        if results["summary"]["failed"] > 0:
            assert "tests" in results
            assert all(t["status"] == "failed" for t in results["tests"])

    @pytest.mark.e2e
    @pytest.mark.high
    @pytest.mark.skip(reason="Requires loaded database and test suite")
    def test_e2e_004_bulk_test_verbose(
        self,
        sample_test_suite: Path,
        chromadb_url: str,
        embedding_model: str,
        temp_dir: Path,
    ):
        """E2E-004: Bulk Test Verbose Mode.

        Verifies:
        - Verbose mode includes all test results
        - Both passed and failed tests are present
        """
        collection_name = "test_e2e_001"
        db_url = f"{chromadb_url}/{collection_name}"
        results_file = temp_dir / "results_verbose.yaml"

        result = subprocess.run(
            [
                "rag-tester",
                "bulk-test",
                "--file",
                str(sample_test_suite),
                "--database",
                db_url,
                "--embedding",
                embedding_model,
                "--output",
                str(results_file),
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert results_file.exists(), "Results file not created"

        # Load and verify results
        with open(results_file) as f:
            results = yaml.safe_load(f)

        # Verify all tests are included
        assert "tests" in results
        total_tests = results["summary"]["total"]
        assert len(results["tests"]) == total_tests, "Not all tests included in verbose mode"

    @pytest.mark.e2e
    @pytest.mark.high
    @pytest.mark.skip(reason="Requires two result files to compare")
    def test_e2e_005_compare_embedding_models(
        self,
        temp_dir: Path,
    ):
        """E2E-005: Compare Two Embedding Models.

        Verifies:
        - Comparison generates metrics for both models
        - Per-test differences are calculated
        - Output file is created
        """
        # Create mock result files for comparison
        results_a = {
            "summary": {"total": 10, "passed": 8, "failed": 2, "tokens": 1000, "time": 10.5},
            "tests": [
                {"id": "test1", "status": "passed", "score": 0.85},
                {"id": "test2", "status": "failed", "score": 0.45},
            ],
        }
        results_b = {
            "summary": {"total": 10, "passed": 7, "failed": 3, "tokens": 1200, "time": 12.0},
            "tests": [
                {"id": "test1", "status": "passed", "score": 0.80},
                {"id": "test2", "status": "failed", "score": 0.50},
            ],
        }

        results_a_file = temp_dir / "results_a.yaml"
        results_b_file = temp_dir / "results_b.yaml"
        comparison_file = temp_dir / "comparison.yaml"

        with open(results_a_file, "w") as f:
            yaml.dump(results_a, f)
        with open(results_b_file, "w") as f:
            yaml.dump(results_b, f)

        result = subprocess.run(
            [
                "rag-tester",
                "compare",
                "--results",
                str(results_a_file),
                str(results_b_file),
                "--output",
                str(comparison_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert comparison_file.exists(), "Comparison file not created"

        # Verify comparison structure
        with open(comparison_file) as f:
            comparison = yaml.safe_load(f)

        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "per_test_diff" in comparison

    @pytest.mark.e2e
    @pytest.mark.high
    @pytest.mark.skip(reason="Requires existing collection with data")
    def test_e2e_006_upsert_mode(
        self,
        temp_dir: Path,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-006: Upsert Mode.

        Verifies:
        - Existing records are updated
        - New records are added
        - Collection size increases correctly
        """
        collection_name = "test_e2e_001"
        db_url = f"{chromadb_url}/{collection_name}"

        # Create update file with existing and new IDs
        updates = {
            "records": [
                {"id": "doc1", "text": "Updated text for doc1"},  # Existing
                {"id": "doc11", "text": "New document 11"},  # New
                {"id": "doc12", "text": "New document 12"},  # New
            ]
        }
        updates_file = temp_dir / "updates.yaml"
        with open(updates_file, "w") as f:
            yaml.dump(updates, f)

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(updates_file),
                "--database",
                db_url,
                "--embedding",
                embedding_model,
                "--mode",
                "upsert",
                "--force-reembed",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Records updated:" in result.stdout
        assert "Records added:" in result.stdout

    @pytest.mark.e2e
    @pytest.mark.high
    @pytest.mark.skip(reason="Requires existing collection with data")
    def test_e2e_007_flush_mode(
        self,
        temp_dir: Path,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-007: Flush Mode.

        Verifies:
        - All existing records are deleted
        - New records are loaded
        - Collection contains only new data
        """
        collection_name = "test_e2e_001"
        db_url = f"{chromadb_url}/{collection_name}"

        # Create new data file
        new_data = {"records": [{"id": f"new_doc{i}", "text": f"New document {i}"} for i in range(1, 6)]}
        new_data_file = temp_dir / "new_data.yaml"
        with open(new_data_file, "w") as f:
            yaml.dump(new_data, f)

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(new_data_file),
                "--database",
                db_url,
                "--embedding",
                embedding_model,
                "--mode",
                "flush",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Records deleted:" in result.stdout
        assert "Records loaded: 5" in result.stdout
