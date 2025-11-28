"""Evaluation framework for SmartDoc Analyst.

This module provides the main evaluation orchestration.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .test_cases import TestCase, TEST_CASES
from .metrics import EvaluationMetrics, calculate_metrics, aggregate_metrics


@dataclass
class EvaluationResult:
    """Result of a single test case evaluation.
    
    Attributes:
        test_case: The test case that was evaluated.
        success: Whether the test passed.
        response: System response.
        metrics: Evaluation metrics.
        error: Error message if failed.
        execution_time_ms: Total execution time.
    """
    test_case: TestCase
    success: bool
    response: str = ""
    metrics: Optional[EvaluationMetrics] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.
        
        Returns:
            Dict: Result as dictionary.
        """
        return {
            "test_case": self.test_case.name,
            "success": self.success,
            "response_preview": self.response[:200] if self.response else "",
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report.
    
    Attributes:
        results: Individual test results.
        aggregate_metrics: Summary metrics.
        timestamp: Evaluation timestamp.
        duration_seconds: Total evaluation duration.
    """
    results: List[EvaluationResult] = field(default_factory=list)
    aggregate_metrics: Optional[EvaluationMetrics] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get evaluation summary.
        
        Returns:
            Dict: Summary statistics.
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        
        by_difficulty = {}
        by_category = {}
        
        for result in self.results:
            diff = result.test_case.difficulty
            cat = result.test_case.category
            
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "passed": 0}
            by_difficulty[diff]["total"] += 1
            if result.success:
                by_difficulty[diff]["passed"] += 1
                
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0}
            by_category[cat]["total"] += 1
            if result.success:
                by_category[cat]["passed"] += 1
                
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "overall_score": self.aggregate_metrics.overall_score() if self.aggregate_metrics else 0,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds
        }
        
    def to_markdown(self) -> str:
        """Generate markdown report.
        
        Returns:
            str: Markdown formatted report.
        """
        lines = [
            "# SmartDoc Analyst Evaluation Report",
            "",
            f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {self.duration_seconds:.1f} seconds",
            "",
            "## Summary",
            ""
        ]
        
        summary = self.summary()
        
        lines.extend([
            f"- **Total Tests:** {summary['total_tests']}",
            f"- **Passed:** {summary['passed']}",
            f"- **Failed:** {summary['failed']}",
            f"- **Pass Rate:** {summary['pass_rate']:.1%}",
            f"- **Overall Score:** {summary['overall_score']:.3f}",
            "",
            "## Results by Difficulty",
            "",
            "| Difficulty | Total | Passed | Rate |",
            "|------------|-------|--------|------|"
        ])
        
        for diff, stats in summary["by_difficulty"].items():
            rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {diff} | {stats['total']} | {stats['passed']} | {rate:.1%} |")
            
        lines.extend([
            "",
            "## Results by Category",
            "",
            "| Category | Total | Passed | Rate |",
            "|----------|-------|--------|------|"
        ])
        
        for cat, stats in summary["by_category"].items():
            rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {cat} | {stats['total']} | {stats['passed']} | {rate:.1%} |")
            
        if self.aggregate_metrics:
            lines.extend([
                "",
                "## Aggregate Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Task Success Rate | {self.aggregate_metrics.task_success_rate:.1%} |",
                f"| Completeness Score | {self.aggregate_metrics.completeness_score:.3f} |",
                f"| Relevance Score | {self.aggregate_metrics.relevance_score:.3f} |",
                f"| Coherence Score | {self.aggregate_metrics.coherence_score:.3f} |",
                f"| Citation Accuracy | {self.aggregate_metrics.citation_accuracy:.3f} |",
                f"| Avg Latency (ms) | {self.aggregate_metrics.response_latency_ms:.0f} |",
                f"| Hallucination Rate | {self.aggregate_metrics.hallucination_rate:.3f} |",
                f"| User Satisfaction | {self.aggregate_metrics.user_satisfaction:.1f}/10 |"
            ])
            
        lines.extend([
            "",
            "## Individual Test Results",
            "",
            "| Test | Difficulty | Category | Status | Score |",
            "|------|------------|----------|--------|-------|"
        ])
        
        for result in self.results:
            status = "✓ Pass" if result.success else "✗ Fail"
            score = result.metrics.overall_score() if result.metrics else 0
            lines.append(
                f"| {result.test_case.name} | {result.test_case.difficulty} | "
                f"{result.test_case.category} | {status} | {score:.3f} |"
            )
            
        return "\n".join(lines)


class EvaluationFramework:
    """Framework for evaluating SmartDoc Analyst.
    
    Provides comprehensive evaluation capabilities including
    running test cases, collecting metrics, and generating reports.
    
    Example:
        >>> framework = EvaluationFramework(analyst)
        >>> report = await framework.run_evaluation()
        >>> print(report.summary())
    """
    
    def __init__(
        self,
        system: Any,  # SmartDocAnalyst instance
        test_cases: Optional[List[TestCase]] = None
    ):
        """Initialize the evaluation framework.
        
        Args:
            system: SmartDocAnalyst instance to evaluate.
            test_cases: Custom test cases (defaults to TEST_CASES).
        """
        self.system = system
        self.test_cases = test_cases or TEST_CASES
        
    async def run_evaluation(
        self,
        test_cases: Optional[List[TestCase]] = None,
        parallel: bool = False
    ) -> EvaluationReport:
        """Run full evaluation suite.
        
        Args:
            test_cases: Specific test cases to run.
            parallel: Run tests in parallel.
            
        Returns:
            EvaluationReport: Complete evaluation report.
        """
        cases = test_cases or self.test_cases
        start_time = datetime.now()
        
        results = []
        
        if parallel:
            # Run tests in parallel
            tasks = [self._run_test(tc) for tc in cases]
            results = await asyncio.gather(*tasks)
        else:
            # Run tests sequentially
            for tc in cases:
                result = await self._run_test(tc)
                results.append(result)
                
        # Calculate aggregate metrics
        valid_metrics = [r.metrics for r in results if r.metrics]
        aggregate = aggregate_metrics(valid_metrics) if valid_metrics else None
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return EvaluationReport(
            results=results,
            aggregate_metrics=aggregate,
            timestamp=start_time,
            duration_seconds=duration
        )
        
    async def _run_test(self, test_case: TestCase) -> EvaluationResult:
        """Run a single test case.
        
        Args:
            test_case: Test case to run.
            
        Returns:
            EvaluationResult: Test result.
        """
        start_time = datetime.now()
        
        try:
            # Ingest test documents
            if test_case.documents:
                self.system.ingest_documents(test_case.documents)
                
            # Run query
            result = await asyncio.wait_for(
                self.system.analyze(test_case.query),
                timeout=test_case.timeout_seconds
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result.get("success"):
                response = result.get("answer", "")
                sources = result.get("sources", [])
                
                # Calculate metrics
                metrics = calculate_metrics(
                    query=test_case.query,
                    response=response,
                    expected_topics=test_case.expected_topics,
                    expected_citations=test_case.expected_citations,
                    sources_found=sources,
                    execution_time_ms=execution_time,
                    success=True
                )
                
                # Determine pass/fail
                passed = (
                    metrics.completeness_score >= 0.5 and
                    metrics.relevance_score >= 0.4 and
                    metrics.overall_score() >= 0.4
                )
                
                return EvaluationResult(
                    test_case=test_case,
                    success=passed,
                    response=response,
                    metrics=metrics,
                    execution_time_ms=execution_time
                )
            else:
                return EvaluationResult(
                    test_case=test_case,
                    success=False,
                    error=result.get("error", "Unknown error"),
                    execution_time_ms=execution_time
                )
                
        except asyncio.TimeoutError:
            return EvaluationResult(
                test_case=test_case,
                success=False,
                error=f"Timeout after {test_case.timeout_seconds}s"
            )
        except Exception as e:
            return EvaluationResult(
                test_case=test_case,
                success=False,
                error=str(e)
            )
        finally:
            # Clear documents between tests
            try:
                self.system.clear_memory()
            except Exception:
                pass
                
    async def run_single_test(self, test_name: str) -> EvaluationResult:
        """Run a single test by name.
        
        Args:
            test_name: Name of test case to run.
            
        Returns:
            EvaluationResult: Test result.
        """
        test_case = next(
            (tc for tc in self.test_cases if tc.name == test_name),
            None
        )
        
        if not test_case:
            return EvaluationResult(
                test_case=TestCase(name=test_name, description="", difficulty="", category="", query=""),
                success=False,
                error=f"Test case '{test_name}' not found"
            )
            
        return await self._run_test(test_case)
        
    async def run_category(self, category: str) -> EvaluationReport:
        """Run all tests in a category.
        
        Args:
            category: Category to run.
            
        Returns:
            EvaluationReport: Category evaluation report.
        """
        cases = [tc for tc in self.test_cases if tc.category == category]
        return await self.run_evaluation(test_cases=cases)
        
    async def run_difficulty(self, difficulty: str) -> EvaluationReport:
        """Run all tests of a difficulty level.
        
        Args:
            difficulty: Difficulty level to run.
            
        Returns:
            EvaluationReport: Difficulty level report.
        """
        cases = [tc for tc in self.test_cases if tc.difficulty == difficulty]
        return await self.run_evaluation(test_cases=cases)
