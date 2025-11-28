"""Orchestrator Agent for SmartDoc Analyst.

The OrchestratorAgent serves as the master coordinator, managing task
delegation to specialized agents and ensuring quality responses.
"""

import asyncio
from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState


class OrchestratorAgent(BaseAgent):
    """Master orchestrator agent for coordinating document analysis.
    
    The OrchestratorAgent is responsible for:
    - Analyzing incoming queries to determine required processing
    - Delegating tasks to appropriate specialized agents
    - Managing parallel and sequential agent execution
    - Aggregating results and ensuring quality
    - Handling retries and error recovery
    
    Attributes:
        agents: Dictionary of available specialized agents.
        max_iterations: Maximum processing iterations.
        parallel_execution: Whether to enable parallel agent execution.
        
    Example:
        >>> orchestrator = OrchestratorAgent(llm=my_llm)
        >>> orchestrator.register_agents(planner=planner, retriever=retriever)
        >>> result = await orchestrator.process(context, "Analyze AI in healthcare")
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        max_iterations: int = 10,
        parallel_execution: bool = True
    ):
        """Initialize the orchestrator agent.
        
        Args:
            llm: Language model interface.
            max_iterations: Maximum processing iterations.
            parallel_execution: Enable parallel agent execution.
        """
        super().__init__(
            name="Orchestrator",
            description="Master coordinator for multi-agent document analysis",
            llm=llm
        )
        self.agents: Dict[str, BaseAgent] = {}
        self.max_iterations = max_iterations
        self.parallel_execution = parallel_execution
        
    def register_agents(self, **agents: BaseAgent) -> None:
        """Register specialized agents for delegation.
        
        Args:
            **agents: Keyword arguments mapping agent names to instances.
            
        Example:
            >>> orchestrator.register_agents(
            ...     planner=PlannerAgent(),
            ...     retriever=RetrieverAgent()
            ... )
        """
        self.agents.update(agents)
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Process a query by coordinating specialized agents.
        
        This method implements the main orchestration logic:
        1. Plan the query using the PlannerAgent
        2. Execute retrieval using the RetrieverAgent
        3. Analyze results using the AnalyzerAgent
        4. Synthesize output using the SynthesizerAgent
        5. Validate quality using the CriticAgent
        
        Args:
            context: Task context with trace information.
            input_data: User query or task specification.
            
        Returns:
            AgentResult: Aggregated results from all agents.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data if isinstance(input_data, str) else str(input_data)
            results = {"query": query, "stages": {}}
            
            # Stage 1: Plan the query decomposition
            if "planner" in self.agents:
                plan_result = await self._execute_agent(
                    "planner", context, {"query": query}
                )
                results["stages"]["planning"] = plan_result
                if not plan_result.success:
                    return AgentResult(
                        success=False,
                        error="Planning stage failed",
                        data=results
                    )
                context.intermediate_results["plan"] = plan_result.data
                
            # Stage 2: Retrieve relevant information
            if "retriever" in self.agents:
                retrieve_result = await self._execute_agent(
                    "retriever", context, {"query": query}
                )
                results["stages"]["retrieval"] = retrieve_result
                context.intermediate_results["retrieved"] = retrieve_result.data
                
            # Stage 3: Analyze the retrieved information
            if "analyzer" in self.agents:
                analyze_result = await self._execute_agent(
                    "analyzer", context, {
                        "query": query,
                        "documents": context.intermediate_results.get("retrieved", {})
                    }
                )
                results["stages"]["analysis"] = analyze_result
                context.intermediate_results["analysis"] = analyze_result.data
                
            # Stage 4: Synthesize the final response
            if "synthesizer" in self.agents:
                synthesize_result = await self._execute_agent(
                    "synthesizer", context, {
                        "query": query,
                        "analysis": context.intermediate_results.get("analysis", {})
                    }
                )
                results["stages"]["synthesis"] = synthesize_result
                context.intermediate_results["synthesis"] = synthesize_result.data
                
            # Stage 5: Quality check with critic
            if "critic" in self.agents:
                critic_result = await self._execute_agent(
                    "critic", context, {
                        "query": query,
                        "response": context.intermediate_results.get("synthesis", {})
                    }
                )
                results["stages"]["critique"] = critic_result
                
                # Retry if quality is insufficient
                if critic_result.data and critic_result.data.get("needs_improvement", False):
                    results["stages"]["retry"] = await self._handle_retry(context, results)
                    
            # Compile final response
            final_response = self._compile_response(results)
            
            self.set_state(AgentState.COMPLETED)
            self.add_to_history({
                "task_id": context.task_id,
                "query": query,
                "success": True
            })
            
            return AgentResult(
                success=True,
                data=final_response,
                metrics=self._collect_metrics(results)
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            self.add_to_history({
                "task_id": context.task_id,
                "query": input_data,
                "success": False,
                "error": str(e)
            })
            return AgentResult(
                success=False,
                error=f"Orchestration failed: {str(e)}",
                data={"partial_results": results if 'results' in dir() else {}}
            )
            
    async def _execute_agent(
        self,
        agent_name: str,
        context: AgentContext,
        input_data: Dict[str, Any]
    ) -> AgentResult:
        """Execute a specific agent with the given input.
        
        Args:
            agent_name: Name of the agent to execute.
            context: Task context.
            input_data: Input data for the agent.
            
        Returns:
            AgentResult: Result from the agent.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return AgentResult(
                success=False,
                error=f"Agent '{agent_name}' not registered"
            )
            
        return await agent.process(context, input_data)
        
    async def _handle_retry(
        self,
        context: AgentContext,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle retry logic when quality check fails.
        
        Args:
            context: Task context.
            results: Current results.
            
        Returns:
            Dict: Retry attempt results.
        """
        retry_results = {}
        critique = results["stages"].get("critique", {})
        
        if critique and hasattr(critique, "suggestions"):
            # Re-run synthesis with critic feedback
            if "synthesizer" in self.agents:
                retry_result = await self._execute_agent(
                    "synthesizer", context, {
                        "query": results["query"],
                        "analysis": context.intermediate_results.get("analysis", {}),
                        "feedback": critique.suggestions
                    }
                )
                retry_results["synthesis_retry"] = retry_result
                context.intermediate_results["synthesis"] = retry_result.data
                
        return retry_results
        
    def _compile_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile the final response from all stage results.
        
        Args:
            results: Results from all processing stages.
            
        Returns:
            Dict: Compiled final response.
        """
        synthesis = results["stages"].get("synthesis")
        
        return {
            "answer": synthesis.data if synthesis and synthesis.success else None,
            "sources": results["stages"].get("retrieval", {}).data if results["stages"].get("retrieval") else [],
            "analysis_summary": results["stages"].get("analysis", {}).data if results["stages"].get("analysis") else {},
            "quality_score": results["stages"].get("critique", {}).data.get("score") if results["stages"].get("critique") and results["stages"].get("critique").data else None,
            "processing_stages": list(results["stages"].keys())
        }
        
    def _collect_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics from all processing stages.
        
        Args:
            results: Results from all stages.
            
        Returns:
            Dict: Aggregated metrics.
        """
        metrics = {
            "stages_completed": len(results.get("stages", {})),
            "total_agents_called": len([
                s for s in results.get("stages", {}).values()
                if hasattr(s, "success") and s.success
            ])
        }
        
        for stage_name, stage_result in results.get("stages", {}).items():
            if hasattr(stage_result, "metrics") and stage_result.metrics:
                metrics[f"{stage_name}_metrics"] = stage_result.metrics
                
        return metrics
        
    def get_capabilities(self) -> List[str]:
        """Return the orchestrator's capabilities.
        
        Returns:
            List[str]: List of capabilities.
        """
        return [
            "query_coordination",
            "agent_delegation",
            "parallel_execution",
            "quality_control",
            "retry_handling",
            "result_aggregation"
        ]
