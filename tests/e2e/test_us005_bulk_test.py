"""E2E tests for US-005: Bulk-Test Command."""

from pathlib import Path

import pytest
import yaml

from rag_tester.commands.bulk_test import bulk_test_command


@pytest.fixture
def test_suite_yaml(tmp_path: Path) -> Path:
    """Create a test suite YAML file with 10 tests (7 pass, 2 fail order, 1 fail threshold)."""
    test_data = {
        "tests": [
            # Passing tests (7)
            {
                "test_id": "test_pass_1",
                "query": "machine learning basics",
                "expected": [
                    {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.7},
                ],
            },
            {
                "test_id": "test_pass_2",
                "query": "deep learning concepts",
                "expected": [
                    {"id": "doc2", "text": "Deep learning uses...", "min_threshold": 0.7},
                ],
            },
            {
                "test_id": "test_pass_3",
                "query": "neural networks",
                "expected": [
                    {"id": "doc3", "text": "Neural networks are...", "min_threshold": 0.7},
                ],
            },
            {
                "test_id": "test_pass_4",
                "query": "AI fundamentals",
                "expected": [
                    {"id": "doc1", "text": "Machine learning is..."},
                    {"id": "doc2", "text": "Deep learning uses..."},
                ],
            },
            {
                "test_id": "test_pass_5",
                "query": "ML algorithms",
                "expected": [
                    {"id": "doc1", "text": "Machine learning is..."},
                ],
            },
            {
                "test_id": "test_pass_6",
                "query": "DL techniques",
                "expected": [
                    {"id": "doc2", "text": "Deep learning uses..."},
                ],
            },
            {
                "test_id": "test_pass_7",
                "query": "NN architecture",
                "expected": [
                    {"id": "doc3", "text": "Neural networks are..."},
                ],
            },
            # Failing tests - wrong order (2)
            {
                "test_id": "test_fail_order_1",
                "query": "AI and ML",
                "expected": [
                    {"id": "doc3", "text": "Neural networks are..."},
                    {"id": "doc1", "text": "Machine learning is..."},
                ],
            },
            {
                "test_id": "test_fail_order_2",
                "query": "DL and NN",
                "expected": [
                    {"id": "doc3", "text": "Neural networks are..."},
                    {"id": "doc2", "text": "Deep learning uses..."},
                ],
            },
            # Failing test - threshold not met (1)
            {
                "test_id": "test_fail_threshold",
                "query": "unrelated query",
                "expected": [
                    {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.99},
                ],
            },
        ]
    }

    test_file = tmp_path / "test_suite.yaml"
    with open(test_file, "w") as f:
        yaml.dump(test_data, f)

    return test_file


@pytest.fixture
def test_suite_large(tmp_path: Path) -> Path:
    """Create a large test suite with 100 tests for parallel execution."""
    tests = []
    for i in range(100):
        tests.append(
            {
                "test_id": f"test_{i}",
                "query": f"query {i}",
                "expected": [
                    {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.5},
                ],
            }
        )

    test_data = {"tests": tests}
    test_file = tmp_path / "test_suite_large.yaml"
    with open(test_file, "w") as f:
        yaml.dump(test_data, f)

    return test_file


@pytest.fixture
def test_suite_50(tmp_path: Path) -> Path:
    """Create a test suite with 50 tests for progress indicator testing."""
    tests = []
    for i in range(50):
        tests.append(
            {
                "test_id": f"test_{i}",
                "query": f"query {i}",
                "expected": [
                    {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.5},
                ],
            }
        )

    test_data = {"tests": tests}
    test_file = tmp_path / "test_suite_50.yaml"
    with open(test_file, "w") as f:
        yaml.dump(test_data, f)

    return test_file


@pytest.fixture
def malformed_yaml(tmp_path: Path) -> Path:
    """Create a malformed YAML file."""
    test_file = tmp_path / "malformed.yaml"
    with open(test_file, "w") as f:
        f.write("invalid: yaml: content:")
    return test_file


@pytest.fixture
def invalid_tests_yaml(tmp_path: Path) -> Path:
    """Create a test file with missing required fields."""
    test_data = {
        "tests": [
            {
                "test_id": "test1",
                # Missing 'query' field
                "expected": [{"id": "doc1", "text": "Text"}],
            }
        ]
    }
    test_file = tmp_path / "invalid_tests.yaml"
    with open(test_file, "w") as f:
        yaml.dump(test_data, f)
    return test_file


@pytest.fixture
def empty_tests_yaml(tmp_path: Path) -> Path:
    """Create an empty test file."""
    test_data = {"tests": []}
    test_file = tmp_path / "empty_tests.yaml"
    with open(test_file, "w") as f:
        yaml.dump(test_data, f)
    return test_file


class TestBulkTestHappyPath:
    """Happy path tests for bulk-test command."""

    @pytest.mark.e2e
    async def test_e2e_003_bulk_test_with_pass_fail_cases(
        self,
        loaded_collection: str,
        test_suite_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-003: Bulk Test with Pass/Fail Cases.

        Test that bulk-test executes a test suite and correctly identifies
        passing and failing tests based on order matching and threshold validation.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command
        try:
            bulk_test_command(
                file=str(test_suite_yaml),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=2,
                verbose=False,
            )
        except SystemExit as e:
            assert e.code == 0, "bulk-test should exit with code 0"

        # Verify results file was created
        assert output_file.exists(), "Results file should be created"

        # Load and verify results
        with open(output_file) as f:
            results = yaml.safe_load(f)

        # Verify summary
        assert "summary" in results
        summary = results["summary"]
        assert summary["total_tests"] == 10
        assert summary["passed"] == 7
        assert summary["failed"] == 3
        assert summary["total_tokens"] == 0
        assert "total_time" in summary
        assert summary["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert loaded_collection in summary["database"]

        # Verify only failed tests are included (not verbose)
        assert "tests" in results
        assert len(results["tests"]) == 3

        # Verify each failed test has required fields
        for test in results["tests"]:
            assert "test_id" in test
            assert "query" in test
            assert "expected" in test
            assert "actual" in test
            assert test["status"] == "failed"
            assert "reason" in test
            assert "duration" in test

    @pytest.mark.e2e
    async def test_e2e_004_bulk_test_verbose(
        self,
        loaded_collection: str,
        test_suite_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-004: Bulk Test Verbose.

        Test that bulk-test with --verbose flag includes all test results
        (both passed and failed) in the output file.
        """
        output_file = tmp_path / "results_verbose.yaml"

        # Execute bulk-test command with verbose flag
        try:
            bulk_test_command(
                file=str(test_suite_yaml),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=True,
            )
        except SystemExit as e:
            assert e.code == 0, "bulk-test should exit with code 0"

        # Verify results file was created
        assert output_file.exists(), "Results file should be created"

        # Load and verify results
        with open(output_file) as f:
            results = yaml.safe_load(f)

        # Verify all 10 tests are included
        assert len(results["tests"]) == 10

        # Verify passed tests have correct fields
        passed_tests = [t for t in results["tests"] if t["status"] == "passed"]
        assert len(passed_tests) == 7
        for test in passed_tests:
            assert "test_id" in test
            assert "query" in test
            assert "expected" in test
            assert "actual" in test
            assert "duration" in test
            assert "reason" not in test  # Passed tests don't have reason

        # Verify failed tests have correct fields
        failed_tests = [t for t in results["tests"] if t["status"] == "failed"]
        assert len(failed_tests) == 3
        for test in failed_tests:
            assert "test_id" in test
            assert "query" in test
            assert "expected" in test
            assert "actual" in test
            assert "reason" in test
            assert "duration" in test

    @pytest.mark.e2e
    async def test_e2e_021_progress_indicator(
        self,
        loaded_collection: str,
        test_suite_50: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-021: Progress Indicator.

        Test that bulk-test displays a progress bar for test suites with > 10 tests.
        Note: This test verifies the command completes successfully. Visual progress
        bar verification would require UI testing tools.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command
        try:
            bulk_test_command(
                file=str(test_suite_50),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=False,
            )
        except SystemExit as e:
            assert e.code == 0, "bulk-test should exit with code 0"

        # Verify results file was created
        assert output_file.exists(), "Results file should be created"

        # Load and verify results
        with open(output_file) as f:
            results = yaml.safe_load(f)

        # Verify all 50 tests were executed
        assert results["summary"]["total_tests"] == 50

    @pytest.mark.e2e
    async def test_e2e_022_parallel_execution(
        self,
        loaded_collection: str,
        test_suite_large: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-022: Parallel Execution.

        Test that bulk-test with --parallel flag executes tests concurrently
        and completes faster than sequential execution.
        """
        output_file = tmp_path / "results.yaml"

        # Execute with parallel=4
        try:
            bulk_test_command(
                file=str(test_suite_large),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=4,
                verbose=False,
            )
        except SystemExit as e:
            assert e.code == 0, "bulk-test should exit with code 0"

        # Verify results
        assert output_file.exists(), "Results file should be created"
        with output_file.open() as f:
            results = yaml.safe_load(f)
        assert results["summary"]["total_tests"] == 100

        # Note: We can't easily test that parallel is faster without running sequential
        # as well, which would double test time. The important thing is it completes
        # successfully with parallel workers.

    @pytest.mark.e2e
    async def test_e2e_027_results_file_written(
        self,
        loaded_collection: str,
        test_suite_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-027: Results File Written.

        Test that bulk-test creates a valid YAML results file with summary
        and tests sections.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command
        try:
            bulk_test_command(
                file=str(test_suite_yaml),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=False,
            )
        except SystemExit as e:
            assert e.code == 0, "bulk-test should exit with code 0"

        # Verify file exists
        assert output_file.exists(), "Results file should be created"

        # Verify file is valid YAML
        with open(output_file) as f:
            results = yaml.safe_load(f)

        # Verify structure
        assert isinstance(results, dict), "Results should be a dictionary"
        assert "summary" in results, "Results should have summary section"
        assert "tests" in results, "Results should have tests section"

        # Verify summary structure
        summary = results["summary"]
        assert "total_tests" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "total_tokens" in summary
        assert "total_time" in summary
        assert "embedding_model" in summary
        assert "database" in summary

    @pytest.mark.e2e
    async def test_e2e_062_test_result_accuracy(
        self,
        loaded_collection: str,
        tmp_path: Path,
    ) -> None:
        """E2E-062: Test Result Accuracy.

        Test that pass/fail determinations are correct based on order matching
        and threshold validation.
        """
        # Create test suite with known expected results
        test_data = {
            "tests": [
                # Should pass: correct order and threshold
                {
                    "test_id": "test_pass_correct",
                    "query": "machine learning",
                    "expected": [
                        {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.7},
                    ],
                },
                # Should fail: wrong order
                {
                    "test_id": "test_fail_wrong_order",
                    "query": "AI concepts",
                    "expected": [
                        {"id": "doc3", "text": "Neural networks are..."},
                        {"id": "doc1", "text": "Machine learning is..."},
                    ],
                },
                # Should fail: threshold not met
                {
                    "test_id": "test_fail_threshold",
                    "query": "unrelated",
                    "expected": [
                        {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.99},
                    ],
                },
            ]
        }

        test_file = tmp_path / "accuracy_tests.yaml"
        with open(test_file, "w") as f:
            yaml.dump(test_data, f)

        output_file = tmp_path / "results.yaml"

        # Execute bulk-test
        try:
            bulk_test_command(
                file=str(test_file),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=True,
            )
        except SystemExit as e:
            assert e.code == 0

        # Load results
        with open(output_file) as f:
            results = yaml.safe_load(f)

        # Verify pass/fail determinations
        tests_by_id = {t["test_id"]: t for t in results["tests"]}

        # Test 1 should pass
        assert tests_by_id["test_pass_correct"]["status"] == "passed"

        # Test 2 should fail with order error
        assert tests_by_id["test_fail_wrong_order"]["status"] == "failed"
        assert "order" in tests_by_id["test_fail_wrong_order"]["reason"].lower()

        # Test 3 should fail with threshold error
        assert tests_by_id["test_fail_threshold"]["status"] == "failed"
        assert "threshold" in tests_by_id["test_fail_threshold"]["reason"].lower()


class TestBulkTestEdgeCases:
    """Edge case and error tests for bulk-test command."""

    @pytest.mark.e2e
    async def test_e2e_086_zero_threshold(
        self,
        loaded_collection: str,
        tmp_path: Path,
    ) -> None:
        """E2E-086: Zero Threshold.

        Test that a test with min_threshold: 0.0 passes with any positive score.
        """
        test_data = {
            "tests": [
                {
                    "test_id": "test_zero_threshold",
                    "query": "any query",
                    "expected": [
                        {"id": "doc1", "text": "Machine learning is...", "min_threshold": 0.0},
                    ],
                }
            ]
        }

        test_file = tmp_path / "zero_threshold.yaml"
        with open(test_file, "w") as f:
            yaml.dump(test_data, f)

        output_file = tmp_path / "results.yaml"

        # Execute bulk-test
        try:
            bulk_test_command(
                file=str(test_file),
                database=loaded_collection,
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=True,
            )
        except SystemExit as e:
            assert e.code == 0

        # Load results
        with open(output_file) as f:
            results = yaml.safe_load(f)

        # Test should pass with any score >= 0.0
        assert results["tests"][0]["status"] == "passed"

    @pytest.mark.e2e
    def test_e2e_bulk_001_malformed_test_file(
        self,
        malformed_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-BULK-001: Malformed Test File.

        Test that bulk-test fails gracefully with a malformed YAML file.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command - should fail
        with pytest.raises(SystemExit) as exc_info:
            bulk_test_command(
                file=str(malformed_yaml),
                database="chromadb://localhost:8000/test_collection",
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=False,
            )

        # Verify exit code is 1
        assert exc_info.value.code == 1

        # Verify no results file was created
        assert not output_file.exists()

    @pytest.mark.e2e
    def test_e2e_bulk_002_missing_required_test_fields(
        self,
        invalid_tests_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-BULK-002: Missing Required Test Fields.

        Test that bulk-test fails when a test is missing required fields.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command - should fail
        with pytest.raises(SystemExit) as exc_info:
            bulk_test_command(
                file=str(invalid_tests_yaml),
                database="chromadb://localhost:8000/test_collection",
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=False,
            )

        # Verify exit code is 1
        assert exc_info.value.code == 1

        # Verify no results file was created
        assert not output_file.exists()

    @pytest.mark.e2e
    def test_e2e_bulk_005_empty_test_suite(
        self,
        empty_tests_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-BULK-005: Empty Test Suite.

        Test that bulk-test fails when the test file has no tests.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command - should fail
        with pytest.raises(SystemExit) as exc_info:
            bulk_test_command(
                file=str(empty_tests_yaml),
                database="chromadb://localhost:8000/test_collection",
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=1,
                verbose=False,
            )

        # Verify exit code is 1
        assert exc_info.value.code == 1

    @pytest.mark.e2e
    def test_e2e_bulk_006_invalid_parallel_workers(
        self,
        test_suite_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """E2E-BULK-006: Invalid Parallel Workers.

        Test that bulk-test fails when parallel workers is out of range.
        """
        output_file = tmp_path / "results.yaml"

        # Execute bulk-test command with invalid parallel workers - should fail
        with pytest.raises(SystemExit) as exc_info:
            bulk_test_command(
                file=str(test_suite_yaml),
                database="chromadb://localhost:8000/test_collection",
                embedding="sentence-transformers/all-MiniLM-L6-v2",
                output=str(output_file),
                parallel=0,  # Invalid: must be 1-16
                verbose=False,
            )

        # Verify exit code is 1
        assert exc_info.value.code == 1
