"""Core module for RAG Tester."""

from rag_tester.core.comparator import (
    ComparatorError,
    calculate_aggregate_metrics,
    identify_per_test_differences,
    validate_result_files_compatible,
)
from rag_tester.core.tester import Tester, TestError, ValidationError

__all__ = [
    "ComparatorError",
    "TestError",
    "Tester",
    "ValidationError",
    "calculate_aggregate_metrics",
    "identify_per_test_differences",
    "validate_result_files_compatible",
]
