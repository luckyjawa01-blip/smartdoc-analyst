"""Document Search Tool for SmartDoc Analyst.

This tool provides semantic search through ingested documents
using vector embeddings and similarity matching.
"""

from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult


class DocumentSearchTool(BaseTool):
    """Semantic search tool for document retrieval.
    
    Uses vector embeddings to find semantically similar documents
    based on user queries. Supports filtering by metadata and
    configurable result counts.
    
    Attributes:
        vector_store: Vector store for document embeddings.
        embedding_model: Model for generating query embeddings.
        default_k: Default number of results to return.
        
    Example:
        >>> tool = DocumentSearchTool(vector_store=my_store)
        >>> result = await tool.execute(
        ...     query="AI in healthcare",
        ...     k=5,
        ...     filters={"domain": "medical"}
        ... )
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        default_k: int = 5
    ):
        """Initialize the document search tool.
        
        Args:
            vector_store: Vector store for document embeddings.
            embedding_model: Model for generating embeddings.
            default_k: Default number of results.
        """
        super().__init__(
            name="document_search",
            description="Search through documents using semantic similarity"
        )
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.default_k = default_k
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute semantic document search.
        
        Args:
            query: Search query text.
            k: Number of results to return.
            filters: Optional metadata filters.
            score_threshold: Minimum similarity score.
            
        Returns:
            ToolResult: Search results with documents.
        """
        query = kwargs.get("query", "")
        k = kwargs.get("k", self.default_k)
        filters = kwargs.get("filters", None)
        score_threshold = kwargs.get("score_threshold", 0.0)
        
        if not query:
            return ToolResult(
                success=False,
                error="Query is required"
            )
            
        if not self.vector_store:
            # Demo mode - return sample results
            return self._demo_search(query, k)
            
        try:
            # Perform similarity search
            if hasattr(self.vector_store, "similarity_search_with_score"):
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filters
                )
                documents = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in results
                    if score >= score_threshold
                ]
            else:
                results = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filters
                )
                documents = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    }
                    for doc in results
                ]
                
            return ToolResult(
                success=True,
                data={
                    "documents": documents,
                    "query": query,
                    "total_results": len(documents)
                },
                metadata={
                    "k": k,
                    "filters_applied": filters is not None
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}"
            )
            
    def _demo_search(self, query: str, k: int) -> ToolResult:
        """Return demo results when no vector store is configured.
        
        Args:
            query: Search query.
            k: Number of results.
            
        Returns:
            ToolResult: Demo search results.
        """
        # Generate contextual demo results based on query keywords
        demo_docs = [
            {
                "content": f"This is a relevant document about {query}. It contains important information that addresses the query topic with detailed analysis and insights.",
                "metadata": {
                    "source": "demo_document_1.pdf",
                    "page": 1,
                    "title": f"Document about {query.split()[0].title()}"
                },
                "score": 0.95
            },
            {
                "content": f"Additional information related to {query}. This document provides supplementary context and supporting details for comprehensive understanding.",
                "metadata": {
                    "source": "demo_document_2.pdf",
                    "page": 3,
                    "title": "Supporting Research"
                },
                "score": 0.87
            },
            {
                "content": f"Background material on topics related to {query}. Includes historical context and foundational concepts necessary for full comprehension.",
                "metadata": {
                    "source": "demo_document_3.pdf",
                    "page": 1,
                    "title": "Background and Context"
                },
                "score": 0.82
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "documents": demo_docs[:k],
                "query": query,
                "total_results": min(k, len(demo_docs)),
                "demo_mode": True
            }
        )
        
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
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": self.default_k
                },
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters"
                },
                "score_threshold": {
                    "type": "number",
                    "description": "Minimum similarity score",
                    "default": 0.0
                }
            },
            "required": ["query"]
        }
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents with content and metadata.
            
        Returns:
            bool: True if successful.
        """
        if not self.vector_store:
            return False
            
        try:
            texts = [doc.get("content", "") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            
            self.vector_store.add_texts(texts, metadatas=metadatas)
            return True
        except Exception:
            return False
