"""Planner Agent for SmartDoc Analyst.

The PlannerAgent handles query decomposition and execution planning
for complex multi-step analysis tasks.
"""

from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState


class PlannerAgent(BaseAgent):
    """Query decomposition and task planning agent.
    
    The PlannerAgent is responsible for:
    - Complex query breakdown
    - Sub-task identification
    - Dependency mapping
    - Execution strategy planning
    
    Attributes:
        max_subtasks: Maximum number of subtasks to generate.
        enable_parallel: Whether to enable parallel task planning.
        
    Example:
        >>> planner = PlannerAgent(llm=my_llm)
        >>> result = await planner.process(context, {
        ...     "query": "Compare AI adoption in healthcare and finance"
        ... })
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        max_subtasks: int = 5,
        enable_parallel: bool = True
    ):
        """Initialize the planner agent.
        
        Args:
            llm: Language model interface.
            max_subtasks: Maximum subtasks to generate.
            enable_parallel: Enable parallel task planning.
        """
        super().__init__(
            name="Planner",
            description="Query decomposition and task planning",
            llm=llm
        )
        self.max_subtasks = max_subtasks
        self.enable_parallel = enable_parallel
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Plan the execution strategy for a query.
        
        Args:
            context: Task context with trace information.
            input_data: Dictionary with query to plan.
            
        Returns:
            AgentResult: Execution plan with subtasks.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            
            # Analyze query complexity
            complexity = self._analyze_complexity(query)
            
            # Decompose into subtasks
            subtasks = await self._decompose_query(query, complexity)
            
            # Build dependency graph
            dependencies = self._build_dependencies(subtasks)
            
            # Generate execution plan
            plan = {
                "original_query": query,
                "complexity": complexity,
                "subtasks": subtasks,
                "dependencies": dependencies,
                "execution_order": self._determine_execution_order(subtasks, dependencies),
                "parallelizable": self._identify_parallel_tasks(subtasks, dependencies),
                "estimated_agents": self._estimate_required_agents(subtasks)
            }
            
            self.set_state(AgentState.COMPLETED)
            self.add_to_history({
                "task_id": context.task_id,
                "query": query,
                "complexity": complexity,
                "subtasks_count": len(subtasks)
            })
            
            return AgentResult(
                success=True,
                data=plan,
                metrics={
                    "complexity_level": complexity,
                    "subtasks_generated": len(subtasks),
                    "parallel_tasks": len(plan["parallelizable"])
                }
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Planning failed: {str(e)}"
            )
            
    def _analyze_complexity(self, query: str) -> str:
        """Analyze the complexity level of a query.
        
        Args:
            query: User query to analyze.
            
        Returns:
            str: Complexity level (simple, medium, complex).
        """
        # Keywords indicating complexity
        complex_keywords = [
            "compare", "analyze", "contrast", "evaluate",
            "multiple", "different", "various", "comprehensive",
            "trend", "correlation", "impact", "relationship"
        ]
        
        medium_keywords = [
            "explain", "describe", "list", "summarize",
            "what", "how", "why", "when"
        ]
        
        query_lower = query.lower()
        
        # Count complexity indicators
        complex_count = sum(1 for kw in complex_keywords if kw in query_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in query_lower)
        
        # Check query length
        word_count = len(query.split())
        
        # Determine complexity
        if complex_count >= 2 or word_count > 30:
            return "complex"
        elif complex_count >= 1 or medium_count >= 2 or word_count > 15:
            return "medium"
        else:
            return "simple"
            
    async def _decompose_query(
        self,
        query: str,
        complexity: str
    ) -> List[Dict[str, Any]]:
        """Decompose query into subtasks.
        
        Args:
            query: Original query.
            complexity: Determined complexity level.
            
        Returns:
            List[Dict]: List of subtasks.
        """
        subtasks = []
        
        if self.llm and complexity != "simple":
            # Use LLM for decomposition
            prompt = f"""Decompose this query into subtasks for a document analysis system:

Query: {query}

Break it down into {self.max_subtasks} or fewer subtasks.
Each subtask should be a specific, actionable step.
Format each as: SUBTASK: <description>"""
            
            try:
                response = await self.llm.generate(prompt)
                if response:
                    subtasks = self._parse_subtasks(response)
            except Exception:
                pass
                
        # Fallback or simple query handling
        if not subtasks:
            subtasks = self._generate_default_subtasks(query, complexity)
            
        return subtasks[:self.max_subtasks]
        
    def _parse_subtasks(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into subtask list.
        
        Args:
            response: LLM response text.
            
        Returns:
            List[Dict]: Parsed subtasks.
        """
        subtasks = []
        lines = response.strip().split("\n")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Parse subtask description
            if "SUBTASK:" in line:
                desc = line.split("SUBTASK:")[-1].strip()
            elif line.startswith(("-", "*", str(i+1))):
                desc = line.lstrip("-*0123456789.) ").strip()
            else:
                desc = line
                
            if desc and len(desc) > 5:
                subtasks.append({
                    "id": f"subtask_{i+1}",
                    "description": desc,
                    "type": self._infer_subtask_type(desc),
                    "status": "pending"
                })
                
        return subtasks
        
    def _generate_default_subtasks(
        self,
        query: str,
        complexity: str
    ) -> List[Dict[str, Any]]:
        """Generate default subtasks based on complexity.
        
        Args:
            query: Original query.
            complexity: Complexity level.
            
        Returns:
            List[Dict]: Default subtasks.
        """
        if complexity == "simple":
            return [
                {
                    "id": "subtask_1",
                    "description": f"Retrieve relevant documents for: {query}",
                    "type": "retrieval",
                    "status": "pending"
                },
                {
                    "id": "subtask_2",
                    "description": "Generate response based on retrieved information",
                    "type": "synthesis",
                    "status": "pending"
                }
            ]
        else:
            return [
                {
                    "id": "subtask_1",
                    "description": f"Search documents for information about: {query}",
                    "type": "retrieval",
                    "status": "pending"
                },
                {
                    "id": "subtask_2",
                    "description": "Search web for current information",
                    "type": "web_search",
                    "status": "pending"
                },
                {
                    "id": "subtask_3",
                    "description": "Analyze and extract key insights",
                    "type": "analysis",
                    "status": "pending"
                },
                {
                    "id": "subtask_4",
                    "description": "Synthesize findings into coherent response",
                    "type": "synthesis",
                    "status": "pending"
                },
                {
                    "id": "subtask_5",
                    "description": "Validate response quality and accuracy",
                    "type": "validation",
                    "status": "pending"
                }
            ]
            
    def _infer_subtask_type(self, description: str) -> str:
        """Infer the type of subtask from description.
        
        Args:
            description: Subtask description.
            
        Returns:
            str: Subtask type.
        """
        desc_lower = description.lower()
        
        if any(kw in desc_lower for kw in ["search", "find", "retrieve", "locate"]):
            if "web" in desc_lower:
                return "web_search"
            return "retrieval"
        elif any(kw in desc_lower for kw in ["analyze", "examine", "evaluate", "assess"]):
            return "analysis"
        elif any(kw in desc_lower for kw in ["synthesize", "combine", "merge", "write", "generate"]):
            return "synthesis"
        elif any(kw in desc_lower for kw in ["validate", "verify", "check", "review"]):
            return "validation"
        elif any(kw in desc_lower for kw in ["calculate", "compute", "measure"]):
            return "calculation"
        else:
            return "general"
            
    def _build_dependencies(
        self,
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Build dependency graph between subtasks.
        
        Args:
            subtasks: List of subtasks.
            
        Returns:
            Dict: Dependency mapping.
        """
        dependencies = {}
        
        # Define type-based dependencies
        type_order = {
            "retrieval": 0,
            "web_search": 0,
            "analysis": 1,
            "calculation": 1,
            "synthesis": 2,
            "validation": 3,
            "general": 1
        }
        
        for subtask in subtasks:
            task_id = subtask["id"]
            task_type = subtask.get("type", "general")
            task_order = type_order.get(task_type, 1)
            
            # Find tasks that must complete before this one
            deps = []
            for other in subtasks:
                if other["id"] == task_id:
                    continue
                other_order = type_order.get(other.get("type", "general"), 1)
                if other_order < task_order:
                    deps.append(other["id"])
                    
            dependencies[task_id] = deps
            
        return dependencies
        
    def _determine_execution_order(
        self,
        subtasks: List[Dict[str, Any]],
        dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """Determine optimal execution order.
        
        Args:
            subtasks: List of subtasks.
            dependencies: Dependency mapping.
            
        Returns:
            List[str]: Ordered task IDs.
        """
        # Topological sort
        order = []
        remaining = {t["id"] for t in subtasks}
        completed = set()
        
        while remaining:
            # Find tasks with all dependencies satisfied
            ready = [
                task_id for task_id in remaining
                if all(dep in completed for dep in dependencies.get(task_id, []))
            ]
            
            if not ready:
                # Break cycle by taking any remaining task
                ready = [list(remaining)[0]]
                
            order.extend(sorted(ready))  # Sort for determinism
            completed.update(ready)
            remaining -= set(ready)
            
        return order
        
    def _identify_parallel_tasks(
        self,
        subtasks: List[Dict[str, Any]],
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Identify tasks that can run in parallel.
        
        Args:
            subtasks: List of subtasks.
            dependencies: Dependency mapping.
            
        Returns:
            List[List[str]]: Groups of parallelizable tasks.
        """
        if not self.enable_parallel:
            return [[t["id"]] for t in subtasks]
            
        # Group tasks by dependency level
        levels = {}
        for subtask in subtasks:
            task_id = subtask["id"]
            deps = dependencies.get(task_id, [])
            
            if not deps:
                level = 0
            else:
                level = max(levels.get(dep, 0) for dep in deps) + 1
                
            levels[task_id] = level
            
        # Group by level
        parallel_groups = {}
        for task_id, level in levels.items():
            if level not in parallel_groups:
                parallel_groups[level] = []
            parallel_groups[level].append(task_id)
            
        return [parallel_groups[i] for i in sorted(parallel_groups.keys())]
        
    def _estimate_required_agents(
        self,
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Estimate which agents are needed for subtasks.
        
        Args:
            subtasks: List of subtasks.
            
        Returns:
            Dict: Agent requirements count.
        """
        type_to_agent = {
            "retrieval": "retriever",
            "web_search": "retriever",
            "analysis": "analyzer",
            "calculation": "analyzer",
            "synthesis": "synthesizer",
            "validation": "critic",
            "general": "orchestrator"
        }
        
        agent_counts = {}
        for subtask in subtasks:
            task_type = subtask.get("type", "general")
            agent = type_to_agent.get(task_type, "orchestrator")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
        return agent_counts
        
    def get_capabilities(self) -> List[str]:
        """Return the planner's capabilities.
        
        Returns:
            List[str]: List of capabilities.
        """
        return [
            "query_decomposition",
            "complexity_analysis",
            "dependency_mapping",
            "parallel_planning",
            "execution_ordering",
            "agent_estimation"
        ]
