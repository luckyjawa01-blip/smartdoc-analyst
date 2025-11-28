"""Base tool class for SmartDoc Analyst.

This module provides the abstract base class for all tools,
defining the common interface and result structure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result returned by a tool after execution.
    
    Attributes:
        success: Whether the tool executed successfully.
        data: The tool's output data.
        error: Error message if unsuccessful.
        execution_time_ms: Execution time in milliseconds.
        metadata: Additional result metadata.
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all SmartDoc Analyst tools.
    
    This class defines the common interface that all tools must implement,
    including execution, validation, and description methods.
    
    Attributes:
        name: Human-readable tool name.
        description: Tool's purpose and capabilities.
        
    Example:
        >>> class MyTool(BaseTool):
        ...     async def execute(self, **kwargs):
        ...         return ToolResult(success=True, data="result")
        ...
        ...     def get_schema(self):
        ...         return {"type": "object", "properties": {...}}
    """
    
    def __init__(self, name: str, description: str):
        """Initialize the base tool.
        
        Args:
            name: Human-readable tool name.
            description: Tool's purpose and capabilities.
        """
        self.name = name
        self.description = description
        self._call_count = 0
        self._total_execution_time_ms = 0.0
        
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.
        
        This is the main method that each tool must implement.
        It receives keyword arguments based on the tool's schema
        and returns a structured result.
        
        Args:
            **kwargs: Tool-specific parameters.
            
        Returns:
            ToolResult: Structured result with success status and data.
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: JSON schema describing expected parameters.
        """
        pass
    
    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters against the schema.
        
        Args:
            **kwargs: Parameters to validate.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        schema = self.get_schema()
        required = schema.get("required", [])
        
        # Check required parameters
        for param in required:
            if param not in kwargs or kwargs[param] is None:
                return False
                
        return True
        
    def get_description(self) -> str:
        """Return the tool's description.
        
        Returns:
            str: Tool description.
        """
        return self.description
        
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Call the tool as a function.
        
        This wrapper method handles timing and call counting.
        
        Args:
            **kwargs: Tool parameters.
            
        Returns:
            ToolResult: Execution result.
        """
        start = datetime.now()
        
        try:
            if not self.validate_input(**kwargs):
                return ToolResult(
                    success=False,
                    error="Invalid input parameters"
                )
                
            result = await self.execute(**kwargs)
            
        except Exception as e:
            result = ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
            
        # Calculate execution time
        execution_time = (datetime.now() - start).total_seconds() * 1000
        result.execution_time_ms = execution_time
        
        # Update statistics
        self._call_count += 1
        self._total_execution_time_ms += execution_time
        
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.
        
        Returns:
            Dict: Usage statistics.
        """
        return {
            "call_count": self._call_count,
            "total_execution_time_ms": self._total_execution_time_ms,
            "avg_execution_time_ms": (
                self._total_execution_time_ms / self._call_count
                if self._call_count > 0 else 0
            )
        }
        
    def __repr__(self) -> str:
        """Return string representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"
