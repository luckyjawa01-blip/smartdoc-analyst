"""Structured logging for SmartDoc Analyst.

This module provides structured logging with contextual
information for debugging and monitoring.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from functools import lru_cache


class SmartDocLogger:
    """Structured logger with context support.
    
    Provides logging with automatic context injection,
    structured output, and configurable formatting.
    
    Attributes:
        name: Logger name.
        level: Logging level.
        context: Default context to include in logs.
        
    Example:
        >>> logger = SmartDocLogger("agent.retriever")
        >>> logger.info("Search completed", extra={
        ...     "query": "AI trends",
        ...     "results": 5
        ... })
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize the logger.
        
        Args:
            name: Logger name (usually module path).
            level: Logging level (DEBUG, INFO, WARNING, ERROR).
            context: Default context for all log messages.
        """
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.context = context or {}
        
        # Create underlying logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.level)
        
        # Add handler if not already present
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.level)
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            
    def _format_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format extra context for logging.
        
        Args:
            extra: Additional context.
            
        Returns:
            Dict: Merged context.
        """
        merged = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            **self.context
        }
        
        if extra:
            merged.update(extra)
            
        return merged
        
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message.
        
        Args:
            message: Log message.
            extra: Additional context.
        """
        self._logger.debug(message, extra={"structured": self._format_extra(extra)})
        
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message.
        
        Args:
            message: Log message.
            extra: Additional context.
        """
        self._logger.info(message, extra={"structured": self._format_extra(extra)})
        
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message.
        
        Args:
            message: Log message.
            extra: Additional context.
        """
        self._logger.warning(message, extra={"structured": self._format_extra(extra)})
        
    def error(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log an error message.
        
        Args:
            message: Log message.
            extra: Additional context.
            exc_info: Include exception info.
        """
        self._logger.error(
            message,
            extra={"structured": self._format_extra(extra)},
            exc_info=exc_info
        )
        
    def exception(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an exception with traceback.
        
        Args:
            message: Log message.
            extra: Additional context.
        """
        self._logger.exception(
            message,
            extra={"structured": self._format_extra(extra)}
        )
        
    def with_context(self, **kwargs: Any) -> "SmartDocLogger":
        """Create a new logger with additional context.
        
        Args:
            **kwargs: Context to add.
            
        Returns:
            SmartDocLogger: New logger with merged context.
        """
        new_context = {**self.context, **kwargs}
        return SmartDocLogger(self.name, logging.getLevelName(self.level), new_context)
        
    def set_level(self, level: str) -> None:
        """Set the logging level.
        
        Args:
            level: New logging level.
        """
        self.level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(self.level)
        
    def add_handler(self, handler: logging.Handler) -> None:
        """Add a custom handler to the logger.
        
        Args:
            handler: Logging handler to add.
        """
        self._logger.addHandler(handler)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output.
    
    Formats log messages with structured context in a
    human-readable format suitable for both development
    and production environments.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        """Initialize the formatter.
        
        Args:
            use_colors: Use ANSI color codes.
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
        
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.
        
        Args:
            record: Log record to format.
            
        Returns:
            str: Formatted log string.
        """
        # Get structured data if available
        structured = getattr(record, 'structured', {})
        
        # Build timestamp
        timestamp = structured.get('timestamp', datetime.now().isoformat())
        
        # Build level string with optional color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            level_str = f"{color}{level:8}{reset}"
        else:
            level_str = f"{level:8}"
            
        # Build main message
        main = f"{timestamp} | {level_str} | {record.name} | {record.getMessage()}"
        
        # Add structured context
        context_items = {
            k: v for k, v in structured.items()
            if k not in ('timestamp', 'logger')
        }
        
        if context_items:
            context_str = " | ".join(f"{k}={v}" for k, v in context_items.items())
            main = f"{main} | {context_str}"
            
        # Add exception info if present
        if record.exc_info:
            main = f"{main}\n{self.formatException(record.exc_info)}"
            
        return main


class JsonFormatter(logging.Formatter):
    """JSON formatter for production logging.
    
    Outputs log messages in JSON format suitable for
    log aggregation systems like ELK or CloudWatch.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            str: JSON formatted log string.
        """
        import json
        
        structured = getattr(record, 'structured', {})
        
        log_data = {
            "timestamp": structured.get('timestamp', datetime.now().isoformat()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **{k: v for k, v in structured.items() if k not in ('timestamp', 'logger')}
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


@lru_cache()
def get_logger(name: str, level: str = "INFO") -> SmartDocLogger:
    """Get a cached logger instance.
    
    Args:
        name: Logger name.
        level: Logging level.
        
    Returns:
        SmartDocLogger: Logger instance.
        
    Example:
        >>> logger = get_logger("smartdoc.agents")
        >>> logger.info("Agent initialized")
    """
    return SmartDocLogger(name, level)


# Configure root logger
def configure_logging(
    level: str = "INFO",
    json_output: bool = False
) -> None:
    """Configure global logging settings.
    
    Args:
        level: Default logging level.
        json_output: Use JSON format for output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root_logger.level)
    
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(StructuredFormatter())
        
    root_logger.addHandler(handler)
