"""Tests for comparator logic."""

import pytest

from rag_tester.core.comparator import (
    ComparatorError,
    calculate_aggregate_metrics,
    identify_per_test_differences,
    validate_result_files_compatible,
)


class TestCalculateAggregateMetrics:
    """Tests for calculate_aggregate_metrics function."""

    def test_calculate_metrics_basic(self) -> None:
        """Test basic metrics calculation."""
        result_data = {
            "summary": {
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "total_tokens": 15000,
                "total_time": 5.2,
                "embedding_model": "openai/text-embedding-3-small",
                "database": "chromadb://localhost:8000/test_collection",
            },
            "tests": [
                {
                    "test_id": "test1",
                    "status": "passed",
                    "actual": [
                        {"id": "doc1", "score": 0.9},
                        {"id": "doc2", "score": 0.85},
                    ],
                },
                {
                    "test_id": "test2",
                    "status": "failed",
                    "actual": [
                        {"id": "doc3", "score": 0.75},
                    ],
                },
            ],
        }

        metrics = calculate_aggregate_metrics(result_data, "openai/text-embedding-3-small")

        assert metrics["name"] == "openai/text-embedding-3-small"
        assert metrics["database"] == "chromadb://localhost:8000/test_collection"
        assert metrics["pass_rate"] == 0.8  # 8/10
        assert metrics["total_tokens"] == 15000
        assert metrics["total_time"] == 5.2
        assert metrics["total_cost"] == 0.0003  # (15000/1M) * 0.02
        assert metrics["cost_per_test"] == 0.00003  # 0.0003 / 10

    def test_calculate_metrics_with_avg_score(self) -> None:
        """Test average score calculation from actual results."""
        result_data = {
            "summary": {
                "total_tests": 3,
                "passed": 2,
                "failed": 1,
                "total_tokens": 0,
                "total_time": 2.0,
                "embedding_model": "local/model",
                "database": "chromadb://localhost:8000/test",
            },
            "tests": [
                {
                    "test_id": "test1",
                    "status": "passed",
                    "actual": [
                        {"id": "doc1", "score": 0.9},
                        {"id": "doc2", "score": 0.8},
                    ],
                },
                {
                    "test_id": "test2",
                    "status": "passed",
                    "actual": [
                        {"id": "doc3", "score": 0.85},
                    ],
                },
                {
                    "test_id": "test3",
                    "status": "failed",
                    "actual": [
                        {"id": "doc4", "score": 0.7},
                    ],
                },
            ],
        }

        metrics = calculate_aggregate_metrics(result_data, "local/model")

        # Average score: (0.9 + 0.8 + 0.85 + 0.7) / 4 = 3.25 / 4 = 0.8125
        assert metrics["avg_score"] == 0.8125

    def test_calculate_metrics_zero_tests(self) -> None:
        """Test metrics calculation with zero tests (edge case)."""
        result_data = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "embedding_model": "local/model",
                "database": "chromadb://localhost:8000/test",
            },
            "tests": [],
        }

        metrics = calculate_aggregate_metrics(result_data, "local/model")

        assert metrics["pass_rate"] == 0.0
        assert metrics["avg_score"] == 0.0
        assert metrics["cost_per_test"] == 0.0

    def test_calculate_metrics_unknown_model_cost(self) -> None:
        """Test cost calculation for unknown model."""
        result_data = {
            "summary": {
                "total_tests": 5,
                "passed": 5,
                "failed": 0,
                "total_tokens": 10000,
                "total_time": 3.0,
                "embedding_model": "custom/my-model",
                "database": "chromadb://localhost:8000/test",
            },
            "tests": [],
        }

        metrics = calculate_aggregate_metrics(result_data, "custom/my-model")

        assert metrics["total_cost"] == 0.0
        assert metrics["cost_per_test"] == 0.0

    def test_calculate_metrics_no_scores(self) -> None:
        """Test metrics calculation when tests have no scores."""
        result_data = {
            "summary": {
                "total_tests": 2,
                "passed": 1,
                "failed": 1,
                "total_tokens": 0,
                "total_time": 1.0,
                "embedding_model": "local/model",
                "database": "chromadb://localhost:8000/test",
            },
            "tests": [
                {
                    "test_id": "test1",
                    "status": "passed",
                    "actual": [],
                },
            ],
        }

        metrics = calculate_aggregate_metrics(result_data, "local/model")

        assert metrics["avg_score"] == 0.0


class TestIdentifyPerTestDifferences:
    """Tests for identify_per_test_differences function."""

    def test_identify_differences_basic(self) -> None:
        """Test identifying differences between two models."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 3},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "actual": [{"id": "doc1", "score": 0.9}],
                        },
                        {
                            "test_id": "test2",
                            "status": "failed",
                            "actual": [{"id": "doc2", "score": 0.7}],
                        },
                        {
                            "test_id": "test3",
                            "status": "passed",
                            "actual": [{"id": "doc3", "score": 0.85}],
                        },
                    ],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 3},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "actual": [{"id": "doc1", "score": 0.88}],
                        },
                        {
                            "test_id": "test2",
                            "status": "passed",
                            "actual": [{"id": "doc2", "score": 0.86}],
                        },
                        {
                            "test_id": "test3",
                            "status": "failed",
                            "actual": [{"id": "doc3", "score": 0.75}],
                        },
                    ],
                },
            ),
        ]

        differences = identify_per_test_differences(results_list)

        # test1: both passed (no diff)
        # test2: model_a failed, model_b passed (diff)
        # test3: model_a passed, model_b failed (diff)
        assert len(differences) == 2

        # Check test2 difference
        test2_diff = next(d for d in differences if d["test_id"] == "test2")
        assert test2_diff["model_a_status"] == "failed"
        assert test2_diff["model_b_status"] == "passed"
        assert test2_diff["model_a_score"] == 0.7
        assert test2_diff["model_b_score"] == 0.86

        # Check test3 difference
        test3_diff = next(d for d in differences if d["test_id"] == "test3")
        assert test3_diff["model_a_status"] == "passed"
        assert test3_diff["model_b_status"] == "failed"

    def test_identify_differences_no_disagreements(self) -> None:
        """Test when all models agree (no differences)."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 2},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "actual": [{"id": "doc1", "score": 0.9}],
                        },
                        {
                            "test_id": "test2",
                            "status": "passed",
                            "actual": [{"id": "doc2", "score": 0.85}],
                        },
                    ],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 2},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "actual": [{"id": "doc1", "score": 0.88}],
                        },
                        {
                            "test_id": "test2",
                            "status": "passed",
                            "actual": [{"id": "doc2", "score": 0.87}],
                        },
                    ],
                },
            ),
        ]

        differences = identify_per_test_differences(results_list)

        assert len(differences) == 0

    def test_identify_differences_with_threshold(self) -> None:
        """Test differences include expected_threshold when present."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 1},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "expected": [{"id": "doc1", "min_threshold": 0.85}],
                            "actual": [{"id": "doc1", "score": 0.9}],
                        },
                    ],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 1},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "failed",
                            "expected": [{"id": "doc1", "min_threshold": 0.85}],
                            "actual": [{"id": "doc1", "score": 0.78}],
                        },
                    ],
                },
            ),
        ]

        differences = identify_per_test_differences(results_list)

        assert len(differences) == 1
        assert differences[0]["expected_threshold"] == 0.85

    def test_identify_differences_three_models(self) -> None:
        """Test differences with three models."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 1},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "actual": [{"id": "doc1", "score": 0.9}],
                        },
                    ],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 1},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "failed",
                            "actual": [{"id": "doc1", "score": 0.7}],
                        },
                    ],
                },
            ),
            (
                "model_c",
                {
                    "summary": {"total_tests": 1},
                    "tests": [
                        {
                            "test_id": "test1",
                            "status": "passed",
                            "actual": [{"id": "doc1", "score": 0.88}],
                        },
                    ],
                },
            ),
        ]

        differences = identify_per_test_differences(results_list)

        assert len(differences) == 1
        assert "model_a_status" in differences[0]
        assert "model_b_status" in differences[0]
        assert "model_c_status" in differences[0]

    def test_identify_differences_sorted_by_test_id(self) -> None:
        """Test that differences are sorted by test_id."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 3},
                    "tests": [
                        {"test_id": "test3", "status": "passed", "actual": []},
                        {"test_id": "test1", "status": "passed", "actual": []},
                        {"test_id": "test2", "status": "passed", "actual": []},
                    ],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 3},
                    "tests": [
                        {"test_id": "test3", "status": "failed", "actual": []},
                        {"test_id": "test1", "status": "failed", "actual": []},
                        {"test_id": "test2", "status": "failed", "actual": []},
                    ],
                },
            ),
        ]

        differences = identify_per_test_differences(results_list)

        assert len(differences) == 3
        assert differences[0]["test_id"] == "test1"
        assert differences[1]["test_id"] == "test2"
        assert differences[2]["test_id"] == "test3"

    def test_identify_differences_single_model(self) -> None:
        """Test with single model (should return empty list)."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 1},
                    "tests": [
                        {"test_id": "test1", "status": "passed", "actual": []},
                    ],
                },
            ),
        ]

        differences = identify_per_test_differences(results_list)

        assert len(differences) == 0


class TestValidateResultFilesCompatible:
    """Tests for validate_result_files_compatible function."""

    def test_validate_compatible_files(self) -> None:
        """Test validation passes for compatible files."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 10},
                    "tests": [],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 10},
                    "tests": [],
                },
            ),
        ]

        # Should not raise
        validate_result_files_compatible(results_list)

    def test_validate_incompatible_test_counts(self) -> None:
        """Test validation fails for different test counts."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 10},
                    "tests": [],
                },
            ),
            (
                "model_b",
                {
                    "summary": {"total_tests": 15},
                    "tests": [],
                },
            ),
        ]

        with pytest.raises(ComparatorError) as exc_info:
            validate_result_files_compatible(results_list)

        assert "Test suites do not match" in str(exc_info.value)
        assert "10 tests" in str(exc_info.value)
        assert "15 tests" in str(exc_info.value)

    def test_validate_single_file(self) -> None:
        """Test validation with single file (should pass)."""
        results_list = [
            (
                "model_a",
                {
                    "summary": {"total_tests": 10},
                    "tests": [],
                },
            ),
        ]

        # Should not raise
        validate_result_files_compatible(results_list)

    def test_validate_three_compatible_files(self) -> None:
        """Test validation with three compatible files."""
        results_list = [
            ("model_a", {"summary": {"total_tests": 20}, "tests": []}),
            ("model_b", {"summary": {"total_tests": 20}, "tests": []}),
            ("model_c", {"summary": {"total_tests": 20}, "tests": []}),
        ]

        # Should not raise
        validate_result_files_compatible(results_list)

    def test_validate_three_incompatible_files(self) -> None:
        """Test validation fails when one of three files is incompatible."""
        results_list = [
            ("model_a", {"summary": {"total_tests": 20}, "tests": []}),
            ("model_b", {"summary": {"total_tests": 20}, "tests": []}),
            ("model_c", {"summary": {"total_tests": 25}, "tests": []}),
        ]

        with pytest.raises(ComparatorError) as exc_info:
            validate_result_files_compatible(results_list)

        assert "Test suites do not match" in str(exc_info.value)
