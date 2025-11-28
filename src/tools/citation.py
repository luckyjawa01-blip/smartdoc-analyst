"""Citation Tool for SmartDoc Analyst.

This tool manages citation tracking and formatting
for document analysis outputs.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from .base_tool import BaseTool, ToolResult


class CitationTool(BaseTool):
    """Citation management and formatting tool.
    
    Tracks citations from various sources and formats them
    according to different academic styles (APA, MLA, Chicago, etc.).
    
    Attributes:
        default_style: Default citation style.
        citations: Dictionary of tracked citations.
        
    Example:
        >>> tool = CitationTool(default_style="apa")
        >>> await tool.execute(
        ...     action="add",
        ...     source_type="document",
        ...     title="AI Research Paper",
        ...     author="Smith, J."
        ... )
    """
    
    def __init__(self, default_style: str = "apa"):
        """Initialize the citation tool.
        
        Args:
            default_style: Default citation format style.
        """
        super().__init__(
            name="citation",
            description="Track and format citations from sources"
        )
        self.default_style = default_style
        self.citations: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute citation management action.
        
        Args:
            action: Action to perform (add, get, format, list, clear).
            **kwargs: Action-specific parameters.
            
        Returns:
            ToolResult: Action result.
        """
        action = kwargs.get("action", "add")
        
        if action == "add":
            return self._add_citation(**kwargs)
        elif action == "get":
            return self._get_citation(kwargs.get("citation_id", ""))
        elif action == "format":
            return self._format_citations(kwargs.get("style", self.default_style))
        elif action == "list":
            return self._list_citations()
        elif action == "clear":
            return self._clear_citations()
        else:
            return ToolResult(
                success=False,
                error=f"Unknown action: {action}"
            )
            
    def _add_citation(self, **kwargs: Any) -> ToolResult:
        """Add a new citation.
        
        Args:
            source_type: Type of source (document, web, book, article).
            title: Source title.
            author: Author name(s).
            url: Source URL (for web sources).
            date: Publication date.
            page: Page number(s).
            publisher: Publisher name.
            
        Returns:
            ToolResult: Add result with citation ID.
        """
        self._counter += 1
        citation_id = f"cite_{self._counter}"
        
        citation = {
            "id": citation_id,
            "source_type": kwargs.get("source_type", "document"),
            "title": kwargs.get("title", "Untitled"),
            "author": kwargs.get("author", "Unknown"),
            "url": kwargs.get("url", ""),
            "date": kwargs.get("date", datetime.now().strftime("%Y")),
            "page": kwargs.get("page", ""),
            "publisher": kwargs.get("publisher", ""),
            "accessed": datetime.now().isoformat(),
            "metadata": kwargs.get("metadata", {})
        }
        
        self.citations[citation_id] = citation
        
        return ToolResult(
            success=True,
            data={
                "citation_id": citation_id,
                "citation": citation
            }
        )
        
    def _get_citation(self, citation_id: str) -> ToolResult:
        """Get a specific citation.
        
        Args:
            citation_id: ID of citation to retrieve.
            
        Returns:
            ToolResult: Citation data.
        """
        if citation_id in self.citations:
            return ToolResult(
                success=True,
                data=self.citations[citation_id]
            )
        else:
            return ToolResult(
                success=False,
                error=f"Citation not found: {citation_id}"
            )
            
    def _list_citations(self) -> ToolResult:
        """List all tracked citations.
        
        Returns:
            ToolResult: List of all citations.
        """
        return ToolResult(
            success=True,
            data={
                "citations": list(self.citations.values()),
                "total": len(self.citations)
            }
        )
        
    def _clear_citations(self) -> ToolResult:
        """Clear all tracked citations.
        
        Returns:
            ToolResult: Clear confirmation.
        """
        count = len(self.citations)
        self.citations.clear()
        self._counter = 0
        
        return ToolResult(
            success=True,
            data={"cleared": count}
        )
        
    def _format_citations(self, style: str) -> ToolResult:
        """Format all citations in the specified style.
        
        Args:
            style: Citation style (apa, mla, chicago, ieee).
            
        Returns:
            ToolResult: Formatted citations.
        """
        formatters = {
            "apa": self._format_apa,
            "mla": self._format_mla,
            "chicago": self._format_chicago,
            "ieee": self._format_ieee
        }
        
        formatter = formatters.get(style.lower(), self._format_apa)
        
        formatted = []
        for citation_id, citation in self.citations.items():
            formatted.append({
                "id": citation_id,
                "formatted": formatter(citation),
                "inline": self._format_inline(citation, style)
            })
            
        return ToolResult(
            success=True,
            data={
                "formatted_citations": formatted,
                "style": style,
                "total": len(formatted)
            }
        )
        
    def _format_apa(self, citation: Dict[str, Any]) -> str:
        """Format citation in APA style.
        
        Args:
            citation: Citation data.
            
        Returns:
            str: APA formatted citation.
        """
        author = citation.get("author", "Unknown")
        year = citation.get("date", "n.d.")
        title = citation.get("title", "Untitled")
        
        source_type = citation.get("source_type", "document")
        
        if source_type == "web":
            url = citation.get("url", "")
            return f"{author} ({year}). {title}. Retrieved from {url}"
        elif source_type == "book":
            publisher = citation.get("publisher", "")
            return f"{author} ({year}). *{title}*. {publisher}."
        else:
            page = citation.get("page", "")
            page_str = f", p. {page}" if page else ""
            return f"{author} ({year}). {title}{page_str}."
            
    def _format_mla(self, citation: Dict[str, Any]) -> str:
        """Format citation in MLA style.
        
        Args:
            citation: Citation data.
            
        Returns:
            str: MLA formatted citation.
        """
        author = citation.get("author", "Unknown")
        title = citation.get("title", "Untitled")
        year = citation.get("date", "n.d.")
        
        source_type = citation.get("source_type", "document")
        
        if source_type == "web":
            url = citation.get("url", "")
            return f'{author}. "{title}." Web. {year}. <{url}>.'
        else:
            publisher = citation.get("publisher", "")
            return f'{author}. "{title}." {publisher}, {year}.'
            
    def _format_chicago(self, citation: Dict[str, Any]) -> str:
        """Format citation in Chicago style.
        
        Args:
            citation: Citation data.
            
        Returns:
            str: Chicago formatted citation.
        """
        author = citation.get("author", "Unknown")
        title = citation.get("title", "Untitled")
        year = citation.get("date", "n.d.")
        publisher = citation.get("publisher", "")
        
        return f'{author}. "{title}." {publisher}, {year}.'
        
    def _format_ieee(self, citation: Dict[str, Any]) -> str:
        """Format citation in IEEE style.
        
        Args:
            citation: Citation data.
            
        Returns:
            str: IEEE formatted citation.
        """
        author = citation.get("author", "Unknown")
        title = citation.get("title", "Untitled")
        year = citation.get("date", "n.d.")
        
        # IEEE uses numbered references
        return f'{author}, "{title}," {year}.'
        
    def _format_inline(self, citation: Dict[str, Any], style: str) -> str:
        """Format inline citation reference.
        
        Args:
            citation: Citation data.
            style: Citation style.
            
        Returns:
            str: Inline citation format.
        """
        author = citation.get("author", "Unknown").split(",")[0]
        year = citation.get("date", "n.d.")
        
        if style.lower() == "ieee":
            return f"[{citation['id'].split('_')[1]}]"
        elif style.lower() == "mla":
            page = citation.get("page", "")
            return f"({author} {page})" if page else f"({author})"
        else:  # APA, Chicago
            return f"({author}, {year})"
            
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: Parameter schema.
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["add", "get", "format", "list", "clear"]
                },
                "citation_id": {
                    "type": "string",
                    "description": "Citation ID (for get action)"
                },
                "source_type": {
                    "type": "string",
                    "description": "Type of source",
                    "enum": ["document", "web", "book", "article"]
                },
                "title": {
                    "type": "string",
                    "description": "Source title"
                },
                "author": {
                    "type": "string",
                    "description": "Author name(s)"
                },
                "url": {
                    "type": "string",
                    "description": "Source URL"
                },
                "date": {
                    "type": "string",
                    "description": "Publication date"
                },
                "style": {
                    "type": "string",
                    "description": "Citation format style",
                    "enum": ["apa", "mla", "chicago", "ieee"]
                }
            },
            "required": ["action"]
        }
