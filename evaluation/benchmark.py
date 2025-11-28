"""Benchmark runner for SmartDoc Analyst.

This module provides performance benchmarking capabilities.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import statistics


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.
    
    Attributes:
        name: Benchmark name.
        iterations: Number of iterations run.
        total_time_seconds: Total benchmark time.
        avg_latency_ms: Average latency per operation.
        min_latency_ms: Minimum latency.
        max_latency_ms: Maximum latency.
        p50_latency_ms: 50th percentile latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        throughput_ops: Operations per second.
        success_rate: Percentage of successful operations.
        errors: List of errors encountered.
    """
    name: str
    iterations: int
    total_time_seconds: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops: float
    success_rate: float
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.
        
        Returns:
            Dict: Result as dictionary.
        """
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 2),
                "min": round(self.min_latency_ms, 2),
                "max": round(self.max_latency_ms, 2),
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2)
            },
            "throughput_ops": round(self.throughput_ops, 2),
            "success_rate": round(self.success_rate, 3),
            "error_count": len(self.errors)
        }
        
    def __str__(self) -> str:
        """Return string representation."""
        return f"""
Benchmark: {self.name}
====================
Iterations: {self.iterations}
Total Time: {self.total_time_seconds:.2f}s
Throughput: {self.throughput_ops:.2f} ops/sec
Success Rate: {self.success_rate:.1%}

Latency (ms):
  Average: {self.avg_latency_ms:.2f}
  Min: {self.min_latency_ms:.2f}
  Max: {self.max_latency_ms:.2f}
  P50: {self.p50_latency_ms:.2f}
  P95: {self.p95_latency_ms:.2f}
  P99: {self.p99_latency_ms:.2f}
"""


class Benchmark:
    """Performance benchmark runner.
    
    Runs various benchmarks to measure system performance
    including latency, throughput, and resource usage.
    
    Example:
        >>> benchmark = Benchmark(analyst)
        >>> result = await benchmark.run_query_benchmark(iterations=100)
        >>> print(result)
    """
    
    def __init__(self, system: Any):
        """Initialize benchmark runner.
        
        Args:
            system: SmartDocAnalyst instance to benchmark.
        """
        self.system = system
        
    async def run_query_benchmark(
        self,
        query: str = "What are the key trends?",
        iterations: int = 10,
        warmup: int = 2
    ) -> BenchmarkResult:
        """Benchmark query processing.
        
        Args:
            query: Query to benchmark.
            iterations: Number of iterations.
            warmup: Warmup iterations (not counted).
            
        Returns:
            BenchmarkResult: Benchmark results.
        """
        latencies = []
        errors = []
        successes = 0
        
        # Warmup
        for _ in range(warmup):
            try:
                await self.system.analyze(query)
            except Exception:
                pass
                
        # Benchmark
        start_time = time.time()
        
        for i in range(iterations):
            iter_start = time.time()
            
            try:
                result = await self.system.analyze(query)
                if result.get("success"):
                    successes += 1
            except Exception as e:
                errors.append(str(e))
                
            latencies.append((time.time() - iter_start) * 1000)
            
        total_time = time.time() - start_time
        
        return self._calculate_result(
            name="query_processing",
            iterations=iterations,
            total_time=total_time,
            latencies=latencies,
            successes=successes,
            errors=errors
        )
        
    async def run_ingestion_benchmark(
        self,
        documents: List[Dict[str, Any]],
        iterations: int = 10
    ) -> BenchmarkResult:
        """Benchmark document ingestion.
        
        Args:
            documents: Documents to ingest.
            iterations: Number of iterations.
            
        Returns:
            BenchmarkResult: Benchmark results.
        """
        latencies = []
        errors = []
        successes = 0
        
        start_time = time.time()
        
        for _ in range(iterations):
            # Clear before each iteration
            self.system.clear_memory()
            
            iter_start = time.time()
            
            try:
                result = self.system.ingest_documents(documents)
                if result.get("added", 0) > 0:
                    successes += 1
            except Exception as e:
                errors.append(str(e))
                
            latencies.append((time.time() - iter_start) * 1000)
            
        total_time = time.time() - start_time
        
        return self._calculate_result(
            name="document_ingestion",
            iterations=iterations,
            total_time=total_time,
            latencies=latencies,
            successes=successes,
            errors=errors
        )
        
    async def run_search_benchmark(
        self,
        query: str = "test query",
        iterations: int = 50
    ) -> BenchmarkResult:
        """Benchmark search operations.
        
        Args:
            query: Search query.
            iterations: Number of iterations.
            
        Returns:
            BenchmarkResult: Benchmark results.
        """
        latencies = []
        errors = []
        successes = 0
        
        start_time = time.time()
        
        for _ in range(iterations):
            iter_start = time.time()
            
            try:
                result = await self.system.search(query)
                successes += 1
            except Exception as e:
                errors.append(str(e))
                
            latencies.append((time.time() - iter_start) * 1000)
            
        total_time = time.time() - start_time
        
        return self._calculate_result(
            name="search",
            iterations=iterations,
            total_time=total_time,
            latencies=latencies,
            successes=successes,
            errors=errors
        )
        
    async def run_concurrent_benchmark(
        self,
        query: str = "What are the key trends?",
        concurrency: int = 5,
        total_requests: int = 20
    ) -> BenchmarkResult:
        """Benchmark concurrent request handling.
        
        Args:
            query: Query to run.
            concurrency: Number of concurrent requests.
            total_requests: Total requests to make.
            
        Returns:
            BenchmarkResult: Benchmark results.
        """
        latencies = []
        errors = []
        successes = 0
        
        async def make_request():
            nonlocal successes
            start = time.time()
            try:
                result = await self.system.analyze(query)
                if result.get("success"):
                    successes += 1
                return (time.time() - start) * 1000, None
            except Exception as e:
                return (time.time() - start) * 1000, str(e)
                
        start_time = time.time()
        
        # Create batches
        for i in range(0, total_requests, concurrency):
            batch_size = min(concurrency, total_requests - i)
            tasks = [make_request() for _ in range(batch_size)]
            results = await asyncio.gather(*tasks)
            
            for latency, error in results:
                latencies.append(latency)
                if error:
                    errors.append(error)
                    
        total_time = time.time() - start_time
        
        return self._calculate_result(
            name=f"concurrent_{concurrency}",
            iterations=total_requests,
            total_time=total_time,
            latencies=latencies,
            successes=successes,
            errors=errors
        )
        
    async def run_full_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run complete benchmark suite.
        
        Returns:
            Dict[str, BenchmarkResult]: All benchmark results.
        """
        results = {}
        
        # Query benchmark
        results["query"] = await self.run_query_benchmark(iterations=5)
        
        # Search benchmark
        results["search"] = await self.run_search_benchmark(iterations=20)
        
        # Ingestion benchmark
        sample_docs = [
            {
                "content": "Sample document content for benchmarking.",
                "metadata": {"source": f"benchmark_{i}.txt"}
            }
            for i in range(5)
        ]
        results["ingestion"] = await self.run_ingestion_benchmark(sample_docs, iterations=5)
        
        # Concurrent benchmark
        results["concurrent"] = await self.run_concurrent_benchmark(
            concurrency=3,
            total_requests=9
        )
        
        return results
        
    def _calculate_result(
        self,
        name: str,
        iterations: int,
        total_time: float,
        latencies: List[float],
        successes: int,
        errors: List[str]
    ) -> BenchmarkResult:
        """Calculate benchmark statistics.
        
        Args:
            name: Benchmark name.
            iterations: Number of iterations.
            total_time: Total time in seconds.
            latencies: List of latencies in ms.
            successes: Number of successes.
            errors: List of errors.
            
        Returns:
            BenchmarkResult: Calculated results.
        """
        if not latencies:
            latencies = [0]
            
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_seconds=total_time,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=sorted_latencies[int(n * 0.5)],
            p95_latency_ms=sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
            p99_latency_ms=sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
            throughput_ops=iterations / total_time if total_time > 0 else 0,
            success_rate=successes / iterations if iterations > 0 else 0,
            errors=errors
        )
        
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult]
    ) -> str:
        """Generate markdown benchmark report.
        
        Args:
            results: Dictionary of benchmark results.
            
        Returns:
            str: Markdown report.
        """
        lines = [
            "# SmartDoc Analyst Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "| Benchmark | Iterations | Throughput | Avg Latency | P95 Latency | Success Rate |",
            "|-----------|------------|------------|-------------|-------------|--------------|"
        ]
        
        for name, result in results.items():
            lines.append(
                f"| {name} | {result.iterations} | "
                f"{result.throughput_ops:.1f} ops/s | "
                f"{result.avg_latency_ms:.0f} ms | "
                f"{result.p95_latency_ms:.0f} ms | "
                f"{result.success_rate:.1%} |"
            )
            
        lines.extend(["", "## Detailed Results", ""])
        
        for name, result in results.items():
            lines.append(f"### {name.title()} Benchmark")
            lines.append("")
            lines.append(f"- **Iterations:** {result.iterations}")
            lines.append(f"- **Total Time:** {result.total_time_seconds:.2f}s")
            lines.append(f"- **Throughput:** {result.throughput_ops:.2f} ops/sec")
            lines.append(f"- **Success Rate:** {result.success_rate:.1%}")
            lines.append("")
            lines.append("**Latency Distribution (ms):**")
            lines.append(f"- Min: {result.min_latency_ms:.2f}")
            lines.append(f"- Avg: {result.avg_latency_ms:.2f}")
            lines.append(f"- P50: {result.p50_latency_ms:.2f}")
            lines.append(f"- P95: {result.p95_latency_ms:.2f}")
            lines.append(f"- P99: {result.p99_latency_ms:.2f}")
            lines.append(f"- Max: {result.max_latency_ms:.2f}")
            lines.append("")
            
        return "\n".join(lines)
