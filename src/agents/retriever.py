"""Retriever Agent for SmartDoc Analyst.

The RetrieverAgent is responsible for information retrieval from
both local document stores and external web sources.
"""

from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState


class RetrieverAgent(BaseAgent):
    """Information retrieval agent for document and web search.
    
    The RetrieverAgent is responsible for:
    - Semantic search through ingested documents
    - Web search for current information
    - Citation tracking and source attribution
    - Result ranking and filtering
    
    Attributes:
        vector_store: Vector store for document embeddings.
        web_search_tool: Tool for web searches.
        top_k: Number of results to retrieve.
        
    Example:
        >>> retriever = RetrieverAgent(vector_store=my_store, llm=my_llm)
        >>> result = await retriever.process(context, {"query": "AI trends"})
    """
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        web_search_tool: Optional[Any] = None,
        llm: Optional[Any] = None,
        top_k: int = 5
    ):
        """Initialize the retriever agent.
        
        Args:
            vector_store: Vector store for document search.
            web_search_tool: Tool for web searches.
            llm: Language model interface.
            top_k: Number of results to retrieve.
        """
        super().__init__(
            name="Retriever",
            description="Information retrieval from documents and web",
            llm=llm
        )
        self.vector_store = vector_store
        self.web_search_tool = web_search_tool
        self.top_k = top_k
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Retrieve relevant information for the query.
        
        Args:
            context: Task context with trace information.
            input_data: Dictionary with query and optional filters.
            
        Returns:
            AgentResult: Retrieved documents and sources.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            filters = input_data.get("filters", {}) if isinstance(input_data, dict) else {}
            
            results = {
                "documents": [],
                "web_results": [],
                "citations": []
            }
            
            # Search vector store for documents
            if self.vector_store:
                doc_results = await self._search_documents(query, filters)
                results["documents"] = doc_results
                results["citations"].extend(self._extract_citations(doc_results))
                
            # Search web for additional context
            if self.web_search_tool and input_data.get("include_web", True):
                web_results = await self._search_web(query)
                results["web_results"] = web_results
                results["citations"].extend(self._extract_web_citations(web_results))
                
            # Rank and filter results
            ranked_results = self._rank_results(results, query)
            
            self.set_state(AgentState.COMPLETED)
            self.add_to_history({
                "task_id": context.task_id,
                "query": query,
                "documents_found": len(results["documents"]),
                "web_results_found": len(results["web_results"])
            })
            
            return AgentResult(
                success=True,
                data=ranked_results,
                metrics={
                    "documents_retrieved": len(results["documents"]),
                    "web_results_retrieved": len(results["web_results"]),
                    "total_citations": len(results["citations"])
                }
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Retrieval failed: {str(e)}"
            )
            
    async def _search_documents(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search the vector store for relevant documents.
        
        Args:
            query: Search query.
            filters: Optional filters for the search.
            
        Returns:
            List[Dict]: Retrieved documents with metadata.
        """
        if not self.vector_store:
            return []
            
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query,
                k=self.top_k,
                filter=filters if filters else None
            )
            
            return [
                {
                    "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    "score": getattr(doc, 'score', None)
                }
                for doc in results
            ]
        except Exception:
            # Return empty list if search fails
            return []
            
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Search the web for relevant information.
        
        Args:
            query: Search query.
            
        Returns:
            List[Dict]: Web search results.
        """
        if not self.web_search_tool:
            return []
            
        try:
            results = await self.web_search_tool.search(query, max_results=self.top_k)
            return results
        except Exception:
            return []
            
    def _extract_citations(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract citation information from documents.
        
        Args:
            documents: List of retrieved documents.
            
        Returns:
            List[Dict]: Citation information.
        """
        citations = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            citations.append({
                "id": f"doc_{i}",
                "source": metadata.get("source", "Unknown"),
                "title": metadata.get("title", "Untitled"),
                "page": metadata.get("page", None),
                "type": "document"
            })
        return citations
        
    def _extract_web_citations(
        self,
        web_results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract citation information from web results.
        
        Args:
            web_results: List of web search results.
            
        Returns:
            List[Dict]: Citation information.
        """
        citations = []
        for i, result in enumerate(web_results, 1):
            citations.append({
                "id": f"web_{i}",
                "source": result.get("url", ""),
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "type": "web"
            })
        return citations
        
    def _rank_results(
        self,
        results: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """Rank and filter combined results.
        
        Args:
            results: Combined retrieval results.
            query: Original query for relevance scoring.
            
        Returns:
            Dict: Ranked and filtered results.
        """
        # Simple ranking by placing documents before web results
        # In production, this would use more sophisticated ranking
        ranked = {
            "documents": sorted(
                results["documents"],
                key=lambda x: x.get("score", 0) if x.get("score") else 0,
                reverse=True
            )[:self.top_k],
            "web_results": results["web_results"][:self.top_k],
            "citations": results["citations"][:self.top_k * 2],
            "query": query
        }
        return ranked
        
    def get_capabilities(self) -> List[str]:
        """Return the retriever's capabilities.
        
        Returns:
            List[str]: List of capabilities.
        """
        return [
            "semantic_document_search",
            "web_search",
            "citation_tracking",
            "result_ranking",
            "multi_source_retrieval"
        ]
