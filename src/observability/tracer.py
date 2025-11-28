"""Distributed tracing for SmartDoc Analyst.

This module provides distributed tracing capabilities
for tracking request flows across agents and services.
"""

import uuid
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator
from threading import local


@dataclass
class Span:
    """A single trace span representing an operation.
    
    Attributes:
        trace_id: Unique identifier for the entire trace.
        span_id: Unique identifier for this span.
        parent_id: ID of the parent span.
        name: Operation name.
        start_time: When the span started.
        end_time: When the span ended.
        attributes: Span attributes.
        events: List of events within the span.
        status: Span status (ok, error).
    """
    trace_id: str
    span_id: str
    name: str
    parent_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute.
        
        Args:
            key: Attribute key.
            value: Attribute value.
        """
        self.attributes[key] = value
        
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event to the span.
        
        Args:
            name: Event name.
            attributes: Event attributes.
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
        
    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set the span status.
        
        Args:
            status: Status (ok, error).
            message: Optional status message.
        """
        self.status = status
        if message:
            self.attributes["status_message"] = message
            
    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now()
        
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds.
        
        Returns:
            Optional[float]: Duration or None if not ended.
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary.
        
        Returns:
            Dict: Span as dictionary.
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms(),
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status
        }


class Tracer:
    """Distributed tracing manager.
    
    Manages trace and span creation, propagation, and collection
    for tracking request flows across the system.
    
    Attributes:
        service_name: Name of the service being traced.
        enabled: Whether tracing is enabled.
        
    Example:
        >>> tracer = Tracer(service_name="smartdoc")
        >>> with tracer.span("process_query") as span:
        ...     span.set_attribute("query", "AI trends")
        ...     result = process(query)
        ...     span.set_attribute("success", True)
    """
    
    def __init__(
        self,
        service_name: str = "smartdoc",
        enabled: bool = True,
        max_spans: int = 1000
    ):
        """Initialize the tracer.
        
        Args:
            service_name: Service name for spans.
            enabled: Enable tracing.
            max_spans: Maximum spans to retain.
        """
        self.service_name = service_name
        self.enabled = enabled
        self.max_spans = max_spans
        
        # Thread-local storage for current span context
        self._context = local()
        
        # Span storage
        self._spans: List[Span] = []
        self._active_traces: Dict[str, List[Span]] = {}
        
    @property
    def id(self) -> Optional[str]:
        """Get current trace ID.
        
        Returns:
            Optional[str]: Current trace ID or None.
        """
        current = self._get_current_span()
        return current.trace_id if current else None
        
    def _get_current_span(self) -> Optional[Span]:
        """Get the current active span.
        
        Returns:
            Optional[Span]: Current span or None.
        """
        return getattr(self._context, 'current_span', None)
        
    def _set_current_span(self, span: Optional[Span]) -> None:
        """Set the current active span.
        
        Args:
            span: Span to set as current.
        """
        self._context.current_span = span
        
    def start_trace(self, name: str) -> Span:
        """Start a new trace.
        
        Args:
            name: Root span name.
            
        Returns:
            Span: Root span of the new trace.
        """
        trace_id = str(uuid.uuid4())
        span = self._create_span(name, trace_id, None)
        self._active_traces[trace_id] = [span]
        return span
        
    def _create_span(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str]
    ) -> Span:
        """Create a new span.
        
        Args:
            name: Span name.
            trace_id: Trace identifier.
            parent_id: Parent span ID.
            
        Returns:
            Span: New span.
        """
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:8],
            name=name,
            parent_id=parent_id
        )
        span.set_attribute("service", self.service_name)
        return span
        
    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Generator[Span, None, None]:
        """Context manager for creating spans.
        
        Args:
            name: Span name.
            attributes: Initial attributes.
            
        Yields:
            Span: The created span.
            
        Example:
            >>> with tracer.span("orchestrator.process") as span:
            ...     span.set_attribute("query", query)
            ...     result = self.process(query)
        """
        if not self.enabled:
            # Return a dummy span if tracing disabled
            dummy = Span(
                trace_id="disabled",
                span_id="disabled",
                name=name
            )
            yield dummy
            return
            
        # Get or create trace
        parent = self._get_current_span()
        
        if parent:
            span = self._create_span(name, parent.trace_id, parent.span_id)
        else:
            span = self.start_trace(name)
            
        # Set initial attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
                
        # Set as current span
        previous_span = self._get_current_span()
        self._set_current_span(span)
        
        try:
            yield span
            span.set_status("ok")
        except Exception as e:
            span.set_status("error", str(e))
            span.add_event("exception", {
                "type": type(e).__name__,
                "message": str(e)
            })
            raise
        finally:
            span.end()
            self._record_span(span)
            self._set_current_span(previous_span)
            
    def _record_span(self, span: Span) -> None:
        """Record a completed span.
        
        Args:
            span: Span to record.
        """
        self._spans.append(span)
        
        # Add to active trace
        if span.trace_id in self._active_traces:
            self._active_traces[span.trace_id].append(span)
            
        # Prune old spans
        if len(self._spans) > self.max_spans:
            self._spans = self._spans[-self.max_spans:]
            
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace.
        
        Args:
            trace_id: Trace identifier.
            
        Returns:
            List[Span]: Spans in the trace.
        """
        return [s for s in self._spans if s.trace_id == trace_id]
        
    def get_recent_traces(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces.
        
        Args:
            n: Number of traces to return.
            
        Returns:
            List[Dict]: Recent trace summaries.
        """
        # Group spans by trace
        traces = {}
        for span in self._spans:
            if span.trace_id not in traces:
                traces[span.trace_id] = []
            traces[span.trace_id].append(span)
            
        # Get most recent traces
        recent = sorted(
            traces.items(),
            key=lambda x: max(s.start_time for s in x[1]),
            reverse=True
        )[:n]
        
        return [
            {
                "trace_id": trace_id,
                "span_count": len(spans),
                "root_span": spans[0].name if spans else None,
                "duration_ms": self._calculate_trace_duration(spans),
                "status": "error" if any(s.status == "error" for s in spans) else "ok"
            }
            for trace_id, spans in recent
        ]
        
    def _calculate_trace_duration(self, spans: List[Span]) -> Optional[float]:
        """Calculate total trace duration.
        
        Args:
            spans: Spans in the trace.
            
        Returns:
            Optional[float]: Duration in milliseconds.
        """
        if not spans:
            return None
            
        start = min(s.start_time for s in spans)
        ends = [s.end_time for s in spans if s.end_time]
        
        if not ends:
            return None
            
        end = max(ends)
        return (end - start).total_seconds() * 1000
        
    def get_span_tree(self, trace_id: str) -> Dict[str, Any]:
        """Get hierarchical span tree for a trace.
        
        Args:
            trace_id: Trace identifier.
            
        Returns:
            Dict: Hierarchical span structure.
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return {}
            
        # Find root spans (no parent)
        roots = [s for s in spans if s.parent_id is None]
        
        def build_tree(parent_span: Span) -> Dict[str, Any]:
            children = [s for s in spans if s.parent_id == parent_span.span_id]
            return {
                **parent_span.to_dict(),
                "children": [build_tree(c) for c in children]
            }
            
        if roots:
            return build_tree(roots[0])
        return {"spans": [s.to_dict() for s in spans]}
        
    def export_json(self) -> List[Dict[str, Any]]:
        """Export all spans as JSON.
        
        Returns:
            List[Dict]: All spans as dictionaries.
        """
        return [span.to_dict() for span in self._spans]
        
    def clear(self) -> int:
        """Clear all recorded spans.
        
        Returns:
            int: Number of spans cleared.
        """
        count = len(self._spans)
        self._spans.clear()
        self._active_traces.clear()
        return count
        
    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics.
        
        Returns:
            Dict: Tracing statistics.
        """
        traces = set(s.trace_id for s in self._spans)
        error_count = sum(1 for s in self._spans if s.status == "error")
        
        durations = [s.duration_ms() for s in self._spans if s.duration_ms() is not None]
        
        return {
            "total_spans": len(self._spans),
            "total_traces": len(traces),
            "error_spans": error_count,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "enabled": self.enabled
        }


# Global tracer instance
_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "smartdoc") -> Tracer:
    """Get or create the global tracer.
    
    Args:
        service_name: Service name for the tracer.
        
    Returns:
        Tracer: Global tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name=service_name)
    return _tracer
