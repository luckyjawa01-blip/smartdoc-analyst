"""Tools module for SmartDoc Analyst.

This module provides seven specialized tools:
- DocumentSearchTool: Semantic search through documents
- WebSearchTool: Web search for current information
- CodeExecutionTool: Safe sandboxed Python execution
- CitationTool: Citation tracking and formatting
- SummarizationTool: Text summarization
- FactCheckerTool: Fact verification
- VisualizationTool: Data visualization generation
"""

from .base_tool import BaseTool, ToolResult
from .document_search import DocumentSearchTool
from .web_search import WebSearchTool
from .code_execution import CodeExecutionTool
from .citation import CitationTool
from .summarization import SummarizationTool
from .fact_checker import FactCheckerTool
from .visualization import VisualizationTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "DocumentSearchTool",
    "WebSearchTool",
    "CodeExecutionTool",
    "CitationTool",
    "SummarizationTool",
    "FactCheckerTool",
    "VisualizationTool",
]
