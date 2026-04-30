"""E2E tests for US-006: Compare Command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rag_tester.rag_tester import app

runner = CliRunner()


@pytest.fixture
def results_model_a(tmp_path: Path) -> Path:
    """Create results file for model A (8/10 pass, avg_score: 0.85)."""
    file = tmp_path / "results_model_a.yaml"
    data = {
        "summary": {
            "total_tests": 10,
            "passed": 8,
            "failed": 2,
            "total_tokens": 0,
            "total_time": 5.2,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "database": "chromadb://localhost:8001/collection_a",
        },
        "tests": [
            {
                "test_id": "test1",
                "status": "passed",
                "actual": [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.85}],
            },
            {
                "test_id": "test2",
                "status": "passed",
                "actual": [{"id": "doc3", "score": 0.88}],
            },
            {
                "test_id": "test3",
                "status": "passed",
                "actual": [{"id": "doc4", "score": 0.87}],
            },
            {
                "test_id": "test4",
                "status": "passed",
                "actual": [{"id": "doc5", "score": 0.82}],
            },
            {
                "test_id": "test5",
                "status": "passed",
                "actual": [{"id": "doc6", "score": 0.86}],
            },
            {
                "test_id": "test6",
                "status": "passed",
                "actual": [{"id": "doc7", "score": 0.84}],
            },
            {
                "test_id": "test7",
                "status": "failed",
                "actual": [{"id": "doc8", "score": 0.72}],
            },
            {
                "test_id": "test8",
                "status": "passed",
                "actual": [{"id": "doc9", "score": 0.89}],
            },
            {
                "test_id": "test9",
                "status": "passed",
                "actual": [{"id": "doc10", "score": 0.83}],
            },
            {
                "test_id": "test10",
                "status": "failed",
                "actual": [{"id": "doc11", "score": 0.74}],
            },
        ],
    }
    with open(file, "w") as f:
        yaml.dump(data, f)
    return file


@pytest.fixture
def results_model_b(tmp_path: Path) -> Path:
    """Create results file for model B (7/10 pass, avg_score: 0.82)."""
    file = tmp_path / "results_model_b.yaml"
    data = {
        "summary": {
            "total_tests": 10,
            "passed": 7,
            "failed": 3,
            "total_tokens": 0,
            "total_time": 6.1,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "database": "chromadb://localhost:8001/collection_b",
        },
        "tests": [
            {
                "test_id": "test1",
                "status": "passed",
                "actual": [{"id": "doc1", "score": 0.88}, {"id": "doc2", "score": 0.80}],
            },
            {
                "test_id": "test2",
                "status": "passed",
                "actual": [{"id": "doc3", "score": 0.85}],
            },
            {
                "test_id": "test3",
                "status": "failed",
                "actual": [{"id": "doc4", "score": 0.62}],
            },
            {
                "test_id": "test4",
                "status": "passed",
                "actual": [{"id": "doc5", "score": 0.81}],
            },
            {
                "test_id": "test5",
                "status": "passed",
                "actual": [{"id": "doc6", "score": 0.84}],
            },
            {
                "test_id": "test6",
                "status": "passed",
                "actual": [{"id": "doc7", "score": 0.83}],
            },
            {
                "test_id": "test7",
                "status": "passed",
                "actual": [{"id": "doc8", "score": 0.88}],
            },
            {
                "test_id": "test8",
                "status": "passed",
                "actual": [{"id": "doc9", "score": 0.87}],
            },
            {
                "test_id": "test9",
                "status": "failed",
                "actual": [{"id": "doc10", "score": 0.76}],
            },
            {
                "test_id": "test10",
                "status": "failed",
                "actual": [{"id": "doc11", "score": 0.70}],
            },
        ],
    }
    with open(file, "w") as f:
        yaml.dump(data, f)
    return file


@pytest.fixture
def results_api_model(tmp_path: Path) -> Path:
    """Create results file for API model with token counts."""
    file = tmp_path / "results_api_model.yaml"
    data = {
        "summary": {
            "total_tests": 10,
            "passed": 8,
            "failed": 2,
            "total_tokens": 15000,
            "total_time": 8.1,
            "embedding_model": "openai/text-embedding-3-small",
            "database": "chromadb://localhost:8001/collection_api",
        },
        "tests": [
            {
                "test_id": f"test{i}",
                "status": "passed" if i <= 8 else "failed",
                "actual": [{"id": f"doc{i}", "score": 0.85 if i <= 8 else 0.70}],
            }
            for i in range(1, 11)
        ],
    }
    with open(file, "w") as f:
        yaml.dump(data, f)
    return file


@pytest.fixture
def results_model_c(tmp_path: Path) -> Path:
    """Create results file for model C."""
    file = tmp_path / "results_model_c.yaml"
    data = {
        "summary": {
            "total_tests": 10,
            "passed": 6,
            "failed": 4,
            "total_tokens": 0,
            "total_time": 7.3,
            "embedding_model": "model-c",
            "database": "chromadb://localhost:8001/collection_c",
        },
        "tests": [
            {
                "test_id": f"test{i}",
                "status": "passed" if i <= 6 else "failed",
                "actual": [{"id": f"doc{i}", "score": 0.82 if i <= 6 else 0.68}],
            }
            for i in range(1, 11)
        ],
    }
    with open(file, "w") as f:
        yaml.dump(data, f)
    return file


class TestE2E005CompareModels:
    """E2E-005: Compare Two Embedding Models."""

    def test_compare_two_models(
        self,
        results_model_a: Path,
        results_model_b: Path,
        tmp_path: Path,
    ) -> None:
        """Compare two models and verify comparison file structure."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                str(results_model_b),
                "--output",
                str(output),
            ],
        )

        # Verify exit code
        assert result.exit_code == 0

        # Verify comparison file created
        assert output.exists()

        # Load and verify structure
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # Verify model_a section
        assert "model_a" in comparison
        model_a = comparison["model_a"]
        assert model_a["pass_rate"] == 0.8
        assert 0.84 <= model_a["avg_score"] <= 0.86  # Allow small variance
        assert model_a["total_tokens"] == 0
        assert model_a["total_time"] == 5.2
        assert model_a["total_cost"] == 0.0

        # Verify model_b section
        assert "model_b" in comparison
        model_b = comparison["model_b"]
        assert model_b["pass_rate"] == 0.7
        assert 0.81 <= model_b["avg_score"] <= 0.83  # Allow small variance
        assert model_b["total_tokens"] == 0
        assert model_b["total_time"] == 6.1
        assert model_b["total_cost"] == 0.0

        # Verify per_test_diff section
        assert "per_test_diff" in comparison
        per_test_diff = comparison["per_test_diff"]
        assert isinstance(per_test_diff, list)
        # Models disagree on test3, test7, test9, test10 (4 differences)
        assert len(per_test_diff) >= 3  # At least 3 differences

        # Verify log output
        assert "Comparison complete" in result.stdout


class TestE2E023CostCalculation:
    """E2E-023: Cost Calculation."""

    def test_cost_calculation(
        self,
        results_api_model: Path,
        results_model_a: Path,
        tmp_path: Path,
    ) -> None:
        """Verify cost calculation for API models."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_api_model),
                str(results_model_a),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Load comparison
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # Verify API model cost
        model_a = comparison["model_a"]
        assert model_a["total_tokens"] == 15000
        assert model_a["total_cost"] == 0.0003  # (15000 / 1M) * $0.02
        assert model_a["cost_per_test"] == 0.00003  # 0.0003 / 10

        # Verify local model has zero cost
        model_b = comparison["model_b"]
        assert model_b["total_cost"] == 0.0
        assert model_b["cost_per_test"] == 0.0


class TestE2E024PerTestDiff:
    """E2E-024: Per-Test Diff."""

    def test_per_test_diff(
        self,
        results_model_a: Path,
        results_model_b: Path,
        tmp_path: Path,
    ) -> None:
        """Verify per-test differences are correctly identified."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                str(results_model_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Load comparison
        with open(output) as f:
            comparison = yaml.safe_load(f)

        per_test_diff = comparison["per_test_diff"]

        # Verify structure of differences
        for diff in per_test_diff:
            assert "test_id" in diff
            assert "model_a_status" in diff
            assert "model_b_status" in diff
            assert "model_a_score" in diff
            assert "model_b_score" in diff

        # Verify diffs are sorted by test_id
        test_ids = [d["test_id"] for d in per_test_diff]
        assert test_ids == sorted(test_ids)


class TestE2E032ComparisonFileGenerated:
    """E2E-032: Comparison File Generated."""

    def test_comparison_file_generated(
        self,
        results_model_a: Path,
        results_model_b: Path,
        tmp_path: Path,
    ) -> None:
        """Verify comparison file is generated with correct structure."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                str(results_model_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Verify file created
        assert output.exists()

        # Verify valid YAML
        with open(output) as f:
            comparison = yaml.safe_load(f)

        assert isinstance(comparison, dict)

        # Verify required sections
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "per_test_diff" in comparison

        # Verify log message
        assert "Comparison written to" in result.stdout


class TestE2ECOMP001MissingResultFile:
    """E2E-COMP-001: Missing Result File."""

    def test_missing_result_file(self, results_model_a: Path, tmp_path: Path) -> None:
        """Verify error when result file is missing."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                "nonexistent.yaml",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Result file not found: nonexistent.yaml" in result.stdout
        assert not output.exists()


class TestE2ECOMP002InvalidResultFormat:
    """E2E-COMP-002: Invalid Result File Format."""

    def test_invalid_result_format(self, results_model_a: Path, tmp_path: Path) -> None:
        """Verify error when result file has invalid format."""
        malformed = tmp_path / "malformed.yaml"
        malformed.write_text("{ invalid yaml content")

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                str(malformed),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Invalid result file format" in result.stdout
        assert not output.exists()


class TestE2ECOMP003IncompatibleResultFiles:
    """E2E-COMP-003: Incompatible Result Files."""

    def test_incompatible_result_files(self, tmp_path: Path) -> None:
        """Verify error when result files are from different test suites."""
        # Create result files with different test counts
        suite_a = tmp_path / "suite_a.yaml"
        suite_b = tmp_path / "suite_b.yaml"

        data_a = {
            "summary": {
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "total_tokens": 0,
                "total_time": 5.0,
                "embedding_model": "model-a",
                "database": "chromadb://localhost:8001/test",
            },
            "tests": [],
        }

        data_b = {
            "summary": {
                "total_tests": 20,
                "passed": 15,
                "failed": 5,
                "total_tokens": 0,
                "total_time": 10.0,
                "embedding_model": "model-b",
                "database": "chromadb://localhost:8001/test",
            },
            "tests": [],
        }

        with open(suite_a, "w") as f:
            yaml.dump(data_a, f)

        with open(suite_b, "w") as f:
            yaml.dump(data_b, f)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(suite_a),
                str(suite_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Test suites do not match" in result.stdout
        assert not output.exists()


class TestE2ECOMP004SingleResultFile:
    """E2E-COMP-004: Single Result File."""

    def test_single_result_file(self, results_model_a: Path, tmp_path: Path) -> None:
        """Verify error when only one result file is provided."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "At least 2 result files required" in result.stdout


class TestE2ECOMP005UnknownModelPricing:
    """E2E-COMP-005: Unknown Model Pricing."""

    def test_unknown_model_pricing(self, results_model_a: Path, tmp_path: Path) -> None:
        """Verify handling of unknown model pricing."""
        custom_model = tmp_path / "custom_model.yaml"
        data = {
            "summary": {
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "total_tokens": 10000,
                "total_time": 5.0,
                "embedding_model": "custom/my-model",
                "database": "chromadb://localhost:8001/test",
            },
            "tests": [],
        }

        with open(custom_model, "w") as f:
            yaml.dump(data, f)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(custom_model),
                str(results_model_a),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Load comparison
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # Verify custom model has zero cost
        model_a = comparison["model_a"]
        assert model_a["total_cost"] == 0.0
        assert model_a["cost_per_test"] == 0.0


class TestE2ECOMP006ZeroTests:
    """E2E-COMP-006: Zero Tests."""

    def test_zero_tests(self, tmp_path: Path) -> None:
        """Verify handling of result files with zero tests."""
        empty_a = tmp_path / "empty_a.yaml"
        empty_b = tmp_path / "empty_b.yaml"

        data = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "embedding_model": "model",
                "database": "chromadb://localhost:8001/test",
            },
            "tests": [],
        }

        with open(empty_a, "w") as f:
            yaml.dump(data, f)

        with open(empty_b, "w") as f:
            yaml.dump(data, f)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(empty_a),
                str(empty_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Load comparison
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # Verify no division by zero errors
        model_a = comparison["model_a"]
        assert model_a["pass_rate"] == 0.0
        assert model_a["avg_score"] == 0.0
        assert model_a["cost_per_test"] == 0.0


class TestE2ECOMP007AllTestsAgree:
    """E2E-COMP-007: All Tests Agree."""

    def test_all_tests_agree(self, tmp_path: Path) -> None:
        """Verify handling when all models agree on all tests."""
        model_a = tmp_path / "model_a.yaml"
        model_b = tmp_path / "model_b.yaml"

        # Both models have identical pass/fail outcomes
        data_template = {
            "summary": {
                "total_tests": 5,
                "passed": 5,
                "failed": 0,
                "total_tokens": 0,
                "total_time": 3.0,
                "database": "chromadb://localhost:8001/test",
            },
            "tests": [
                {
                    "test_id": f"test{i}",
                    "status": "passed",
                    "actual": [{"id": f"doc{i}", "score": 0.85}],
                }
                for i in range(1, 6)
            ],
        }

        data_a = {**data_template, "summary": {**data_template["summary"], "embedding_model": "model-a"}}
        data_b = {**data_template, "summary": {**data_template["summary"], "embedding_model": "model-b"}}

        with open(model_a, "w") as f:
            yaml.dump(data_a, f)

        with open(model_b, "w") as f:
            yaml.dump(data_b, f)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(model_a),
                str(model_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Load comparison
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # Verify no differences
        per_test_diff = comparison["per_test_diff"]
        assert len(per_test_diff) == 0


class TestE2ECOMP008ThreeModelsComparison:
    """E2E-COMP-008: Three Models Comparison."""

    def test_three_models_comparison(
        self,
        results_model_a: Path,
        results_model_b: Path,
        results_model_c: Path,
        tmp_path: Path,
    ) -> None:
        """Verify comparison with three models."""
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(results_model_a),
                str(results_model_b),
                str(results_model_c),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Load comparison
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # Verify all three models present
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "model_c" in comparison

        # Verify per_test_diff shows disagreements across all three
        per_test_diff = comparison["per_test_diff"]
        for diff in per_test_diff:
            # Each diff should have status and score for all three models
            assert "model_a_status" in diff
            assert "model_b_status" in diff
            assert "model_c_status" in diff
