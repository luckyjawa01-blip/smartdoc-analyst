"""Base agent class for SmartDoc Analyst.

This module provides the abstract base class for all agents,
defining the common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class AgentState(Enum):
    """Enumeration of possible agent states.
    
    Attributes:
        IDLE: Agent is ready and waiting for tasks.
        RUNNING: Agent is currently processing a task.
        WAITING: Agent is waiting for external input.
        COMPLETED: Agent has completed its task.
        ERROR: Agent encountered an error.
    """
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Context information passed between agents.
    
    Attributes:
        task_id: Unique identifier for the current task.
        trace_id: Distributed tracing identifier.
        query: Original user query.
        intermediate_results: Results from other agents.
        metadata: Additional context metadata.
        start_time: Task start timestamp.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Result returned by an agent after processing.
    
    Attributes:
        success: Whether the agent completed successfully.
        data: The agent's output data.
        error: Error message if unsuccessful.
        metrics: Performance metrics for the task.
        suggestions: Suggestions for improvement or next steps.
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all SmartDoc Analyst agents.
    
    This class defines the common interface and shared functionality
    that all specialized agents must implement.
    
    Attributes:
        name: Human-readable agent name.
        description: Agent's role and capabilities.
        state: Current agent state.
        tools: List of tools available to the agent.
        
    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def process(self, context, input_data):
        ...         return AgentResult(success=True, data="processed")
        ...
        ...     def get_capabilities(self):
        ...         return ["processing"]
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: Optional[List[Any]] = None,
        llm: Optional[Any] = None
    ):
        """Initialize the base agent.
        
        Args:
            name: Human-readable agent name.
            description: Agent's role and capabilities.
            tools: List of tools available to the agent.
            llm: Language model interface for the agent.
        """
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.tools = tools or []
        self.llm = llm
        self._history: List[Dict[str, Any]] = []
        
    @abstractmethod
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Process a task and return the result.
        
        This is the main method that each agent must implement.
        It receives context information and input data, processes
        them according to the agent's specialization, and returns
        a structured result.
        
        Args:
            context: Task context with trace IDs and intermediate results.
            input_data: Agent-specific input data.
            
        Returns:
            AgentResult: Structured result with success status and data.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return a list of the agent's capabilities.
        
        Returns:
            List[str]: List of capability strings.
        """
        pass
    
    def set_state(self, state: AgentState) -> None:
        """Update the agent's state.
        
        Args:
            state: New agent state.
        """
        self.state = state
        
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the agent's history.
        
        Args:
            entry: History entry with task details.
        """
        entry["timestamp"] = datetime.now().isoformat()
        self._history.append(entry)
        
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the agent's task history.
        
        Returns:
            List[Dict]: List of historical task entries.
        """
        return self._history.copy()
    
    def clear_history(self) -> None:
        """Clear the agent's task history."""
        self._history.clear()
        
    def __repr__(self) -> str:
        """Return string representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', state={self.state.value})"
