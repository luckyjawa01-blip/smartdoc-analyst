"""Evaluation module for SmartDoc Analyst.

This module provides comprehensive evaluation capabilities:
- Framework: Main evaluation orchestration
- Metrics: Evaluation metric calculations
- TestCases: 20+ test cases for system validation
- Benchmark: Performance benchmarking
"""

from .framework import EvaluationFramework
from .metrics import EvaluationMetrics, calculate_metrics
from .test_cases import TEST_CASES, TestCase
from .benchmark import Benchmark, BenchmarkResult

__all__ = [
    "EvaluationFramework",
    "EvaluationMetrics",
    "calculate_metrics",
    "TEST_CASES",
    "TestCase",
    "Benchmark",
    "BenchmarkResult",
]
