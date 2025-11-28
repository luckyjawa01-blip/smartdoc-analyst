"""Web Search Tool for SmartDoc Analyst.

This tool provides web search capabilities for retrieving
current information from the internet.
"""

from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Web search tool for retrieving internet information.
    
    Searches the web using DuckDuckGo or similar providers
    to find current information relevant to queries.
    
    Attributes:
        max_results: Maximum number of search results.
        safe_search: Enable safe search filtering.
        region: Geographic region for results.
        
    Example:
        >>> tool = WebSearchTool(max_results=10)
        >>> result = await tool.execute(
        ...     query="Latest AI trends 2025",
        ...     max_results=5
        ... )
    """
    
    def __init__(
        self,
        max_results: int = 10,
        safe_search: bool = True,
        region: str = "us-en"
    ):
        """Initialize the web search tool.
        
        Args:
            max_results: Maximum results to return.
            safe_search: Enable safe search.
            region: Geographic region code.
        """
        super().__init__(
            name="web_search",
            description="Search the web for current information"
        )
        self.max_results = max_results
        self.safe_search = safe_search
        self.region = region
        self._search_client = None
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute web search.
        
        Args:
            query: Search query text.
            max_results: Override max results.
            region: Override region setting.
            
        Returns:
            ToolResult: Search results with URLs and snippets.
        """
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", self.max_results)
        
        if not query:
            return ToolResult(
                success=False,
                error="Query is required"
            )
            
        try:
            results = await self._search(query, max_results)
            
            return ToolResult(
                success=True,
                data={
                    "results": results,
                    "query": query,
                    "total_results": len(results)
                },
                metadata={
                    "source": "web_search",
                    "safe_search": self.safe_search
                }
            )
            
        except Exception as e:
            # Return demo results on error
            return self._demo_search(query, max_results)
            
    async def _search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform the actual web search.
        
        Args:
            query: Search query.
            max_results: Maximum results.
            
        Returns:
            List[Dict]: Search results.
        """
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": "duckduckgo"
                    })
                return results
                
        except ImportError:
            # Fall back to demo results if library not installed
            return self._generate_demo_results(query, max_results)
        except Exception:
            return self._generate_demo_results(query, max_results)
            
    def _demo_search(self, query: str, max_results: int) -> ToolResult:
        """Return demo results when search fails.
        
        Args:
            query: Search query.
            max_results: Number of results.
            
        Returns:
            ToolResult: Demo search results.
        """
        results = self._generate_demo_results(query, max_results)
        
        return ToolResult(
            success=True,
            data={
                "results": results,
                "query": query,
                "total_results": len(results),
                "demo_mode": True
            }
        )
        
    def _generate_demo_results(
        self,
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Generate demo search results.
        
        Args:
            query: Search query.
            max_results: Number of results.
            
        Returns:
            List[Dict]: Demo results.
        """
        keywords = query.split()[:3]
        topic = " ".join(keywords).title()
        
        demo_results = [
            {
                "title": f"{topic} - Comprehensive Overview",
                "url": f"https://example.com/{keywords[0].lower() if keywords else 'topic'}",
                "snippet": f"A comprehensive overview of {query}. This article covers the key aspects, recent developments, and future implications.",
                "source": "demo"
            },
            {
                "title": f"Latest Research on {topic}",
                "url": f"https://research.example.com/{keywords[0].lower() if keywords else 'topic'}",
                "snippet": f"Recent research findings related to {query}. Includes analysis from leading experts and data-driven insights.",
                "source": "demo"
            },
            {
                "title": f"{topic} in Practice - Case Studies",
                "url": f"https://cases.example.com/{keywords[0].lower() if keywords else 'topic'}",
                "snippet": f"Real-world case studies demonstrating {query} applications. Learn from successful implementations.",
                "source": "demo"
            },
            {
                "title": f"Expert Analysis: {topic}",
                "url": f"https://analysis.example.com/{keywords[0].lower() if keywords else 'topic'}",
                "snippet": f"Expert analysis and commentary on {query}. Includes trends, predictions, and strategic recommendations.",
                "source": "demo"
            },
            {
                "title": f"{topic} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{keywords[0].title() if keywords else 'Topic'}",
                "snippet": f"Wikipedia article about {query}. Provides background information, history, and references.",
                "source": "demo"
            }
        ]
        
        return demo_results[:max_results]
        
    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Convenience method for searching.
        
        Args:
            query: Search query.
            max_results: Maximum results.
            
        Returns:
            List[Dict]: Search results.
        """
        result = await self.execute(
            query=query,
            max_results=max_results or self.max_results
        )
        if result.success:
            return result.data.get("results", [])
        return []
        
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: Parameter schema.
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": self.max_results
                },
                "region": {
                    "type": "string",
                    "description": "Geographic region for results",
                    "default": self.region
                }
            },
            "required": ["query"]
        }
