"""Message definitions for SmartDoc Analyst.

This module provides standard message formats for
inter-agent communication.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class MessageType(Enum):
    """Types of messages in the A2A protocol.
    
    Attributes:
        TASK: Task assignment message.
        RESULT: Task result message.
        ERROR: Error notification message.
        FEEDBACK: Quality feedback message.
        STATUS: Status update message.
        ACK: Acknowledgment message.
    """
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    FEEDBACK = "feedback"
    STATUS = "status"
    ACK = "ack"


@dataclass
class AgentMessage:
    """Standard A2A message format.
    
    Base class for all inter-agent messages providing
    common fields for routing and tracking.
    
    Attributes:
        id: Unique message identifier.
        from_agent: Sender agent name.
        to_agent: Recipient agent name.
        message_type: Type of message.
        content: Message payload.
        metadata: Additional message metadata.
        timestamp: Message creation timestamp.
        correlation_id: ID for request-response matching.
        priority: Message priority (0-10, higher is more urgent).
        
    Example:
        >>> msg = AgentMessage(
        ...     from_agent="orchestrator",
        ...     to_agent="retriever",
        ...     message_type=MessageType.TASK,
        ...     content={"query": "AI trends"}
        ... )
    """
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.
        
        Returns:
            Dict: Message as dictionary.
        """
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            AgentMessage: Reconstructed message.
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            priority=data.get("priority", 5)
        )
        
    def reply(
        self,
        content: Any,
        message_type: Optional[MessageType] = None
    ) -> "AgentMessage":
        """Create a reply to this message.
        
        Args:
            content: Reply content.
            message_type: Reply message type.
            
        Returns:
            AgentMessage: Reply message.
        """
        return AgentMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=message_type or MessageType.RESULT,
            content=content,
            correlation_id=self.correlation_id,
            metadata={
                "reply_to": self.id,
                **self.metadata
            }
        )


@dataclass
class TaskMessage(AgentMessage):
    """Task assignment message.
    
    Specialized message for assigning tasks to agents.
    
    Attributes:
        task_type: Type of task to perform.
        parameters: Task parameters.
        deadline: Optional task deadline.
        retry_count: Number of retries allowed.
    """
    task_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    retry_count: int = 3
    
    def __post_init__(self):
        """Initialize message type."""
        self.message_type = MessageType.TASK
        if self.parameters:
            self.content = {
                "task_type": self.task_type,
                "parameters": self.parameters
            }
            
    @classmethod
    def create(
        cls,
        from_agent: str,
        to_agent: str,
        task_type: str,
        parameters: Dict[str, Any],
        deadline: Optional[datetime] = None,
        priority: int = 5
    ) -> "TaskMessage":
        """Create a task message.
        
        Args:
            from_agent: Sender agent.
            to_agent: Target agent.
            task_type: Type of task.
            parameters: Task parameters.
            deadline: Optional deadline.
            priority: Task priority.
            
        Returns:
            TaskMessage: New task message.
        """
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.TASK,
            content={"task_type": task_type, "parameters": parameters},
            task_type=task_type,
            parameters=parameters,
            deadline=deadline,
            priority=priority
        )


@dataclass
class ResultMessage(AgentMessage):
    """Task result message.
    
    Specialized message for returning task results.
    
    Attributes:
        success: Whether task succeeded.
        data: Result data.
        execution_time_ms: Execution time in milliseconds.
    """
    success: bool = True
    data: Any = None
    execution_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize message type."""
        self.message_type = MessageType.RESULT
        self.content = {
            "success": self.success,
            "data": self.data,
            "execution_time_ms": self.execution_time_ms
        }
        
    @classmethod
    def from_task(
        cls,
        task_message: TaskMessage,
        success: bool,
        data: Any,
        execution_time_ms: float = 0.0
    ) -> "ResultMessage":
        """Create result message from task.
        
        Args:
            task_message: Original task message.
            success: Whether task succeeded.
            data: Result data.
            execution_time_ms: Execution time.
            
        Returns:
            ResultMessage: Result message.
        """
        return cls(
            from_agent=task_message.to_agent,
            to_agent=task_message.from_agent,
            message_type=MessageType.RESULT,
            content={"success": success, "data": data, "execution_time_ms": execution_time_ms},
            correlation_id=task_message.correlation_id,
            success=success,
            data=data,
            execution_time_ms=execution_time_ms,
            metadata={"task_id": task_message.id}
        )


@dataclass
class ErrorMessage(AgentMessage):
    """Error notification message.
    
    Specialized message for reporting errors.
    
    Attributes:
        error_code: Error code identifier.
        error_message: Human-readable error message.
        recoverable: Whether the error is recoverable.
        details: Additional error details.
    """
    error_code: str = "UNKNOWN"
    error_message: str = ""
    recoverable: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize message type."""
        self.message_type = MessageType.ERROR
        self.content = {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "recoverable": self.recoverable,
            "details": self.details
        }
        
    @classmethod
    def from_exception(
        cls,
        from_agent: str,
        to_agent: str,
        exception: Exception,
        correlation_id: Optional[str] = None
    ) -> "ErrorMessage":
        """Create error message from exception.
        
        Args:
            from_agent: Agent that encountered error.
            to_agent: Agent to notify.
            exception: The exception.
            correlation_id: Original message correlation ID.
            
        Returns:
            ErrorMessage: Error message.
        """
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.ERROR,
            content={
                "error_code": type(exception).__name__,
                "error_message": str(exception),
                "recoverable": True
            },
            correlation_id=correlation_id or str(uuid.uuid4()),
            error_code=type(exception).__name__,
            error_message=str(exception)
        )


@dataclass
class FeedbackMessage(AgentMessage):
    """Quality feedback message.
    
    Specialized message for providing quality feedback.
    
    Attributes:
        rating: Quality rating (0-10).
        suggestions: List of improvement suggestions.
        issues: List of identified issues.
    """
    rating: float = 5.0
    suggestions: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize message type."""
        self.message_type = MessageType.FEEDBACK
        self.content = {
            "rating": self.rating,
            "suggestions": self.suggestions,
            "issues": self.issues
        }
        
    @classmethod
    def create_feedback(
        cls,
        from_agent: str,
        to_agent: str,
        rating: float,
        suggestions: list = None,
        issues: list = None,
        correlation_id: Optional[str] = None
    ) -> "FeedbackMessage":
        """Create a feedback message.
        
        Args:
            from_agent: Feedback provider.
            to_agent: Feedback recipient.
            rating: Quality rating.
            suggestions: Improvement suggestions.
            issues: Identified issues.
            correlation_id: Related message ID.
            
        Returns:
            FeedbackMessage: Feedback message.
        """
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.FEEDBACK,
            content={
                "rating": rating,
                "suggestions": suggestions or [],
                "issues": issues or []
            },
            correlation_id=correlation_id or str(uuid.uuid4()),
            rating=rating,
            suggestions=suggestions or [],
            issues=issues or []
        )
