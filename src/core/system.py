"""Main SmartDoc Analyst system.

This module provides the main SmartDocAnalyst class that
integrates all components into a cohesive system.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..agents import (
    OrchestratorAgent,
    RetrieverAgent,
    AnalyzerAgent,
    SynthesizerAgent,
    CriticAgent,
    PlannerAgent,
    AgentContext
)
from ..tools import (
    DocumentSearchTool,
    WebSearchTool,
    CodeExecutionTool,
    CitationTool,
    SummarizationTool,
    FactCheckerTool,
    VisualizationTool
)
from ..memory import MemoryManager
from ..observability import get_logger, metrics, get_tracer
from ..protocols import MessageBus, A2AProtocol
from .llm_interface import GeminiInterface, LLMInterface
from .safety import SafetyGuard


class SmartDocAnalyst:
    """SmartDoc Analyst - Multi-Agent Document Analysis System.
    
    The main system class that orchestrates document analysis
    using multiple specialized agents, tools, and memory systems.
    
    Features:
    - Six specialized agents (Orchestrator, Retriever, Analyzer, 
      Synthesizer, Critic, Planner)
    - Seven tools (Document Search, Web Search, Code Execution,
      Citation, Summarization, Fact Checker, Visualization)
    - Three-tier memory (Working, Episodic, Semantic)
    - Full observability (Logging, Metrics, Tracing)
    - A2A protocol for inter-agent communication
    - Safety guards for input validation
    
    Attributes:
        settings: Application settings.
        llm: Language model interface.
        memory: Memory manager.
        orchestrator: Main orchestrator agent.
        
    Example:
        >>> analyst = SmartDocAnalyst(api_key="your-api-key")
        >>> analyst.ingest_documents([
        ...     {"content": "AI is transforming healthcare...", 
        ...      "metadata": {"source": "ai_report.pdf"}}
        ... ])
        >>> result = await analyst.analyze("What are the key trends in AI?")
        >>> print(result["answer"])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
        llm: Optional[LLMInterface] = None
    ):
        """Initialize SmartDoc Analyst.
        
        Args:
            api_key: Gemini API key. If not provided, will look
                    for SMARTDOC_GEMINI_API_KEY environment variable.
            settings: Custom settings. Defaults to environment-based settings.
            llm: Custom LLM interface. Defaults to GeminiInterface.
        """
        # Load settings
        self.settings = settings or get_settings()
        
        # Initialize logger
        self.logger = get_logger("smartdoc.system", self.settings.log_level)
        self.logger.info("Initializing SmartDoc Analyst", extra={
            "model": self.settings.model_name,
            "version": "1.0.0"
        })
        
        # Initialize tracer
        self.tracer = get_tracer("smartdoc")
        
        # Initialize LLM
        effective_api_key = api_key or self.settings.gemini_api_key
        self.llm = llm or GeminiInterface(
            api_key=effective_api_key,
            model_name=self.settings.model_name,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens
        )
        
        # Initialize safety guard
        self.safety_guard = SafetyGuard(
            max_input_length=self.settings.max_input_length,
            rate_limit_rpm=self.settings.rate_limit_rpm
        )
        
        # Initialize memory
        self.memory = MemoryManager(
            working_memory_size=100,
            vector_store_path=self.settings.vector_store_path
        )
        
        # Initialize tools
        self._init_tools()
        
        # Initialize agents
        self._init_agents()
        
        # Initialize message bus for A2A
        self.message_bus = MessageBus()
        self._register_agents_with_bus()
        
        self.logger.info("SmartDoc Analyst initialized successfully")
        
    def _init_tools(self) -> None:
        """Initialize all tools."""
        self.tools = {
            "document_search": DocumentSearchTool(
                vector_store=self.memory.get_vector_store()
            ),
            "web_search": WebSearchTool(),
            "code_execution": CodeExecutionTool(),
            "citation": CitationTool(),
            "summarization": SummarizationTool(llm=self.llm),
            "fact_checker": FactCheckerTool(llm=self.llm),
            "visualization": VisualizationTool()
        }
        
    def _init_agents(self) -> None:
        """Initialize all agents."""
        # Create specialized agents
        self.planner = PlannerAgent(llm=self.llm)
        self.retriever = RetrieverAgent(
            vector_store=self.memory.get_vector_store(),
            web_search_tool=self.tools["web_search"],
            llm=self.llm
        )
        self.analyzer = AnalyzerAgent(
            llm=self.llm,
            code_executor=self.tools["code_execution"],
            fact_checker=self.tools["fact_checker"]
        )
        self.synthesizer = SynthesizerAgent(llm=self.llm)
        self.critic = CriticAgent(llm=self.llm)
        
        # Create orchestrator and register agents
        self.orchestrator = OrchestratorAgent(
            llm=self.llm,
            max_iterations=self.settings.max_agent_iterations,
            parallel_execution=self.settings.parallel_agents
        )
        self.orchestrator.register_agents(
            planner=self.planner,
            retriever=self.retriever,
            analyzer=self.analyzer,
            synthesizer=self.synthesizer,
            critic=self.critic
        )
        
    def _register_agents_with_bus(self) -> None:
        """Register agents with the message bus."""
        agents = {
            "orchestrator": self.orchestrator,
            "planner": self.planner,
            "retriever": self.retriever,
            "analyzer": self.analyzer,
            "synthesizer": self.synthesizer,
            "critic": self.critic
        }
        
        for name, agent in agents.items():
            self.message_bus.subscribe(
                name,
                lambda msg, a=agent: a.process(
                    AgentContext(trace_id=msg.correlation_id),
                    msg.content
                )
            )
            
    def ingest_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ingest documents into the system.
        
        Args:
            documents: List of documents with content and metadata.
                Each document should have:
                - content: Text content
                - metadata: Dict with source, title, etc.
                
        Returns:
            Dict: Ingestion results with counts and IDs.
            
        Example:
            >>> analyst.ingest_documents([
            ...     {
            ...         "content": "AI is transforming healthcare...",
            ...         "metadata": {"source": "report.pdf", "title": "AI Report"}
            ...     }
            ... ])
        """
        self.logger.info(f"Ingesting {len(documents)} documents")
        
        with self.tracer.span("ingest_documents") as span:
            span.set_attribute("document_count", len(documents))
            
            validated = []
            rejected = []
            
            for doc in documents:
                # Validate document
                result = self.safety_guard.validate_document(doc)
                
                if result.valid:
                    validated.append({
                        "content": result.sanitized or doc.get("content", ""),
                        "metadata": doc.get("metadata", {})
                    })
                else:
                    rejected.append({
                        "document": doc.get("metadata", {}).get("source", "unknown"),
                        "issues": result.issues
                    })
                    
            # Add to vector store
            added_ids = self.memory.add_documents(validated)
            
            # Store ingestion event
            self.memory.store_episode(
                f"ingestion_{datetime.now().isoformat()}",
                {
                    "documents_added": len(validated),
                    "documents_rejected": len(rejected),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            metrics.increment("documents_ingested", len(validated))
            
            span.set_attribute("documents_added", len(validated))
            span.set_attribute("documents_rejected", len(rejected))
            
        return {
            "added": len(validated),
            "rejected": len(rejected),
            "document_ids": added_ids,
            "rejected_details": rejected
        }
        
    async def analyze(
        self,
        query: str,
        include_web_search: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze documents and answer a query.
        
        This is the main method for querying the system. It:
        1. Validates the input
        2. Plans the analysis approach
        3. Retrieves relevant information
        4. Analyzes the content
        5. Synthesizes a response
        6. Validates quality
        
        Args:
            query: User's question or analysis request.
            include_web_search: Whether to include web search results.
            user_id: Optional user ID for rate limiting.
            
        Returns:
            Dict: Analysis results including:
                - answer: Main response text
                - sources: List of cited sources
                - analysis: Detailed analysis data
                - quality_score: Response quality score
                - execution_time_ms: Processing time
                
        Example:
            >>> result = await analyst.analyze(
            ...     "What are the main trends in AI healthcare?",
            ...     include_web_search=True
            ... )
            >>> print(result["answer"])
        """
        start_time = datetime.now()
        
        with self.tracer.span("analyze", {"query": query[:100]}) as span:
            # Rate limiting
            if user_id and not self.safety_guard.rate_limit(user_id):
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "rate_limit_status": self.safety_guard.get_rate_limit_status(user_id)
                }
                
            # Validate input
            validation = self.safety_guard.validate_input(query)
            if not validation.valid:
                self.logger.warning("Invalid input rejected", extra={
                    "issues": validation.issues,
                    "risk_score": validation.risk_score
                })
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "issues": validation.issues
                }
                
            # Use sanitized query
            safe_query = validation.sanitized or query
            
            # Add to working memory
            self.memory.add_to_context(
                f"Query: {safe_query}",
                metadata={"type": "query"},
                importance=0.9
            )
            
            # Create context
            context = AgentContext(
                query=safe_query,
                metadata={
                    "include_web_search": include_web_search,
                    "user_id": user_id
                }
            )
            
            self.logger.info("Processing query", extra={
                "query": safe_query[:50],
                "task_id": context.task_id
            })
            
            # Process through orchestrator
            result = await self.orchestrator.process(context, safe_query)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record metrics
            metrics.timing("query_latency_ms", execution_time)
            metrics.increment("queries_processed")
            
            if result.success:
                metrics.increment("queries_successful")
            else:
                metrics.increment("queries_failed")
                
            span.set_attribute("success", result.success)
            span.set_attribute("execution_time_ms", execution_time)
            
            # Sanitize output
            if result.success and result.data:
                answer = result.data.get("answer", {})
                if isinstance(answer, dict):
                    response_text = answer.get("response", "")
                else:
                    response_text = str(answer)
                    
                sanitized_response = self.safety_guard.sanitize_output(response_text)
                
                return {
                    "success": True,
                    "answer": sanitized_response,
                    "sources": result.data.get("sources", []),
                    "analysis": result.data.get("analysis_summary", {}),
                    "quality_score": result.data.get("quality_score"),
                    "processing_stages": result.data.get("processing_stages", []),
                    "execution_time_ms": execution_time,
                    "task_id": context.task_id
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Analysis failed",
                    "execution_time_ms": execution_time,
                    "task_id": context.task_id
                }
                
    async def search(
        self,
        query: str,
        k: int = 5,
        include_web: bool = False
    ) -> List[Dict[str, Any]]:
        """Search documents without full analysis.
        
        A lightweight search that returns relevant documents
        without the full analysis pipeline.
        
        Args:
            query: Search query.
            k: Number of results to return.
            include_web: Include web search results.
            
        Returns:
            List[Dict]: Search results with content and metadata.
        """
        with self.tracer.span("search", {"query": query[:50]}):
            # Search documents
            doc_results = self.memory.search_documents(query, k=k)
            
            # Optionally include web results
            if include_web:
                web_results = await self.tools["web_search"].search(query, max_results=k)
                return {
                    "documents": doc_results,
                    "web_results": web_results
                }
                
            return {"documents": doc_results}
            
    async def summarize(
        self,
        text: str,
        max_length: int = 200
    ) -> str:
        """Summarize text content.
        
        Args:
            text: Text to summarize.
            max_length: Maximum summary length in words.
            
        Returns:
            str: Summary text.
        """
        result = await self.tools["summarization"].execute(
            text=text,
            max_length=max_length
        )
        
        if result.success:
            return result.data.get("summary", "")
        return ""
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dict: System statistics including memory, metrics, and traces.
        """
        return {
            "memory": self.memory.get_stats(),
            "metrics": metrics.get_all_metrics(),
            "traces": self.tracer.get_stats(),
            "tools": {
                name: tool.get_stats()
                for name, tool in self.tools.items()
            }
        }
        
    def clear_memory(self) -> Dict[str, int]:
        """Clear all memory stores.
        
        Returns:
            Dict: Count of cleared items per store.
        """
        return self.memory.clear_all()
        
    def get_context(self) -> Dict[str, Any]:
        """Get current context for debugging.
        
        Returns:
            Dict: Current system context.
        """
        return self.memory.get_context_for_llm()


def main():
    """Main entry point for CLI usage."""
    import sys
    
    print("SmartDoc Analyst - Multi-Agent Document Analysis System")
    print("=" * 55)
    print()
    print("Usage:")
    print("  from smartdoc import SmartDocAnalyst")
    print("  analyst = SmartDocAnalyst(api_key='your-key')")
    print("  result = await analyst.analyze('Your question here')")
    print()
    print("For more information, see the documentation.")


if __name__ == "__main__":
    main()
