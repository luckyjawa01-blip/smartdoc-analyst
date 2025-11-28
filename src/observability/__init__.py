"""Observability module for SmartDoc Analyst.

This module provides the full observability stack:
- Logger: Structured logging with context
- Metrics: Performance metrics collection
- Tracer: Distributed tracing for request flows
"""

from .logger import get_logger, SmartDocLogger
from .metrics import MetricsCollector, metrics
from .tracer import Tracer, Span, get_tracer

__all__ = [
    "get_logger",
    "SmartDocLogger",
    "MetricsCollector",
    "metrics",
    "Tracer",
    "Span",
    "get_tracer",
]
