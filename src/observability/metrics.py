"""Metrics collection for SmartDoc Analyst.

This module provides metrics collection and reporting
for monitoring system performance and health.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from threading import Lock


@dataclass
class MetricValue:
    """A single metric measurement.
    
    Attributes:
        name: Metric name.
        value: Metric value.
        timestamp: When the metric was recorded.
        labels: Additional metric labels.
    """
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collector for application metrics.
    
    Collects various types of metrics including counters,
    gauges, histograms, and timing measurements.
    
    Attributes:
        prefix: Prefix for all metric names.
        enabled: Whether metrics collection is enabled.
        
    Example:
        >>> metrics = MetricsCollector(prefix="smartdoc")
        >>> metrics.increment("queries_total")
        >>> metrics.timing("query_latency_ms", 150)
        >>> metrics.gauge("active_agents", 3)
    """
    
    def __init__(
        self,
        prefix: str = "smartdoc",
        enabled: bool = True
    ):
        """Initialize the metrics collector.
        
        Args:
            prefix: Prefix for metric names.
            enabled: Enable metrics collection.
        """
        self.prefix = prefix
        self.enabled = enabled
        
        # Thread-safe storage
        self._lock = Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._last_values: Dict[str, MetricValue] = {}
        
    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix.
        
        Args:
            name: Base metric name.
            
        Returns:
            str: Full metric name.
        """
        return f"{self.prefix}_{name}" if self.prefix else name
        
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric.
        
        Args:
            name: Metric name.
            value: Amount to increment.
            labels: Optional labels.
        """
        if not self.enabled:
            return
            
        full_name = self._full_name(name)
        label_key = self._label_key(full_name, labels)
        
        with self._lock:
            self._counters[label_key] += value
            self._last_values[full_name] = MetricValue(
                name=full_name,
                value=self._counters[label_key],
                labels=labels or {}
            )
            
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Metric name.
            value: Gauge value.
            labels: Optional labels.
        """
        if not self.enabled:
            return
            
        full_name = self._full_name(name)
        label_key = self._label_key(full_name, labels)
        
        with self._lock:
            self._gauges[label_key] = value
            self._last_values[full_name] = MetricValue(
                name=full_name,
                value=value,
                labels=labels or {}
            )
            
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram observation.
        
        Args:
            name: Metric name.
            value: Observed value.
            labels: Optional labels.
        """
        if not self.enabled:
            return
            
        full_name = self._full_name(name)
        label_key = self._label_key(full_name, labels)
        
        with self._lock:
            self._histograms[label_key].append(value)
            # Keep only last 1000 observations
            if len(self._histograms[label_key]) > 1000:
                self._histograms[label_key] = self._histograms[label_key][-1000:]
                
    def timing(
        self,
        name: str,
        value_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timing measurement.
        
        Args:
            name: Metric name.
            value_ms: Time in milliseconds.
            labels: Optional labels.
        """
        if not self.enabled:
            return
            
        full_name = self._full_name(name)
        label_key = self._label_key(full_name, labels)
        
        with self._lock:
            self._timings[label_key].append(value_ms)
            # Keep only last 1000 observations
            if len(self._timings[label_key]) > 1000:
                self._timings[label_key] = self._timings[label_key][-1000:]
                
    def record(self, metrics_dict: Dict[str, Any]) -> None:
        """Record multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric name to value.
            
        Example:
            >>> metrics.record({
            ...     "query_latency_ms": 150,
            ...     "tokens_used": 500,
            ...     "agent_calls": 3
            ... })
        """
        for name, value in metrics_dict.items():
            if name.endswith("_total") or name.endswith("_count"):
                self.increment(name, value)
            elif name.endswith("_ms") or name.endswith("_seconds"):
                self.timing(name, value)
            else:
                self.gauge(name, value)
                
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing code blocks.
        
        Args:
            name: Metric name.
            labels: Optional labels.
            
        Returns:
            Timer context manager.
            
        Example:
            >>> with metrics.timer("operation_duration_ms"):
            ...     perform_operation()
        """
        return Timer(self, name, labels)
        
    def _label_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a unique key for metric with labels.
        
        Args:
            name: Metric name.
            labels: Metric labels.
            
        Returns:
            str: Unique key.
        """
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
        
    def get_counter(self, name: str) -> float:
        """Get current counter value.
        
        Args:
            name: Metric name.
            
        Returns:
            float: Counter value.
        """
        full_name = self._full_name(name)
        with self._lock:
            return self._counters.get(full_name, 0.0)
            
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value.
        
        Args:
            name: Metric name.
            
        Returns:
            Optional[float]: Gauge value or None.
        """
        full_name = self._full_name(name)
        with self._lock:
            return self._gauges.get(full_name)
            
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics.
        
        Args:
            name: Metric name.
            
        Returns:
            Dict: Statistics (count, sum, avg, min, max, p50, p95, p99).
        """
        full_name = self._full_name(name)
        with self._lock:
            values = self._histograms.get(full_name, [])
            
        if not values:
            return {"count": 0}
            
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "sum": sum(sorted_values),
            "avg": sum(sorted_values) / count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1],
            "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1]
        }
        
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """Get timing statistics.
        
        Args:
            name: Metric name.
            
        Returns:
            Dict: Timing statistics in milliseconds.
        """
        full_name = self._full_name(name)
        with self._lock:
            values = self._timings.get(full_name, [])
            
        if not values:
            return {"count": 0}
            
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "total_ms": sum(sorted_values),
            "avg_ms": sum(sorted_values) / count,
            "min_ms": sorted_values[0],
            "max_ms": sorted_values[-1],
            "p50_ms": sorted_values[int(count * 0.5)],
            "p95_ms": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1],
            "p99_ms": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1]
        }
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics.
        
        Returns:
            Dict: All metrics and their values.
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k.replace(f"{self.prefix}_", ""))
                    for k in self._histograms.keys()
                },
                "timings": {
                    k: self.get_timing_stats(k.replace(f"{self.prefix}_", ""))
                    for k in self._timings.keys()
                }
            }
            
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timings.clear()
            self._last_values.clear()
            
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            str: Prometheus-formatted metrics.
        """
        lines = []
        
        with self._lock:
            # Export counters
            for name, value in self._counters.items():
                lines.append(f"{name} {value}")
                
            # Export gauges
            for name, value in self._gauges.items():
                lines.append(f"{name} {value}")
                
        return "\n".join(lines)


class Timer:
    """Context manager for timing code blocks.
    
    Example:
        >>> with Timer(metrics, "operation_ms"):
        ...     do_something()
    """
    
    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Initialize the timer.
        
        Args:
            collector: Metrics collector.
            name: Metric name.
            labels: Optional labels.
        """
        self.collector = collector
        self.name = name
        self.labels = labels
        self._start_time: Optional[float] = None
        
    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record metric."""
        if self._start_time is not None:
            elapsed_ms = (time.time() - self._start_time) * 1000
            self.collector.timing(self.name, elapsed_ms, self.labels)


# Global metrics instance
metrics = MetricsCollector()
