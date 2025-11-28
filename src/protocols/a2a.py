"""Agent-to-Agent Protocol for SmartDoc Analyst.

This module provides the A2A communication protocol
for inter-agent message passing and coordination.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from .messages import AgentMessage, MessageType


class MessageBus:
    """Central message bus for agent communication.
    
    Provides publish-subscribe messaging between agents
    with support for direct messaging and broadcasting.
    
    Attributes:
        subscribers: Agent message handlers.
        message_history: History of messages.
        
    Example:
        >>> bus = MessageBus()
        >>> bus.subscribe("retriever", handler_func)
        >>> await bus.send(message)
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize the message bus.
        
        Args:
            max_history: Maximum messages to retain in history.
        """
        self.max_history = max_history
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_history: List[AgentMessage] = []
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._message_counts: Dict[str, int] = defaultdict(int)
        
    def subscribe(
        self,
        agent_name: str,
        handler: Callable[[AgentMessage], Any]
    ) -> None:
        """Subscribe an agent to receive messages.
        
        Args:
            agent_name: Name of the subscribing agent.
            handler: Message handler function.
        """
        self._subscribers[agent_name].append(handler)
        
    def unsubscribe(self, agent_name: str) -> bool:
        """Unsubscribe an agent from messages.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            bool: True if unsubscribed.
        """
        if agent_name in self._subscribers:
            del self._subscribers[agent_name]
            return True
        return False
        
    async def send(
        self,
        message: AgentMessage,
        wait_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send a message to an agent.
        
        Args:
            message: Message to send.
            wait_response: Whether to wait for response.
            timeout: Response timeout in seconds.
            
        Returns:
            Optional[AgentMessage]: Response message if waiting.
        """
        # Record message
        self._record_message(message)
        
        # Set up response waiting if needed
        if wait_response:
            future = asyncio.get_event_loop().create_future()
            self._pending_responses[message.correlation_id] = future
            
        # Deliver to subscribers
        handlers = self._subscribers.get(message.to_agent, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message)
                else:
                    result = handler(message)
                    
                # If handler returns a response message
                if isinstance(result, AgentMessage):
                    self._record_message(result)
                    
                    # Resolve pending response
                    if result.correlation_id in self._pending_responses:
                        self._pending_responses[result.correlation_id].set_result(result)
                        del self._pending_responses[result.correlation_id]
                        
            except Exception as e:
                # Send error message back
                error_msg = AgentMessage(
                    from_agent=message.to_agent,
                    to_agent=message.from_agent,
                    message_type=MessageType.ERROR,
                    content={"error": str(e)},
                    correlation_id=message.correlation_id
                )
                self._record_message(error_msg)
                
                if message.correlation_id in self._pending_responses:
                    self._pending_responses[message.correlation_id].set_result(error_msg)
                    del self._pending_responses[message.correlation_id]
                    
        # Wait for response if requested
        if wait_response and message.correlation_id in self._pending_responses:
            try:
                response = await asyncio.wait_for(
                    self._pending_responses[message.correlation_id],
                    timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                del self._pending_responses[message.correlation_id]
                return None
                
        return None
        
    async def broadcast(
        self,
        message: AgentMessage,
        exclude: Optional[Set[str]] = None
    ) -> int:
        """Broadcast message to all subscribers.
        
        Args:
            message: Message to broadcast.
            exclude: Agents to exclude.
            
        Returns:
            int: Number of agents notified.
        """
        exclude = exclude or set()
        count = 0
        
        for agent_name in self._subscribers:
            if agent_name not in exclude:
                broadcast_msg = AgentMessage(
                    from_agent=message.from_agent,
                    to_agent=agent_name,
                    message_type=message.message_type,
                    content=message.content,
                    correlation_id=message.correlation_id,
                    metadata=message.metadata
                )
                await self.send(broadcast_msg)
                count += 1
                
        return count
        
    def _record_message(self, message: AgentMessage) -> None:
        """Record a message in history.
        
        Args:
            message: Message to record.
        """
        self._message_history.append(message)
        self._message_counts[message.from_agent] += 1
        
        # Prune history if needed
        if len(self._message_history) > self.max_history:
            self._message_history = self._message_history[-self.max_history:]
            
    def get_history(
        self,
        agent_name: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get message history.
        
        Args:
            agent_name: Filter by agent.
            message_type: Filter by message type.
            limit: Maximum messages to return.
            
        Returns:
            List[AgentMessage]: Message history.
        """
        messages = self._message_history
        
        if agent_name:
            messages = [
                m for m in messages
                if m.from_agent == agent_name or m.to_agent == agent_name
            ]
            
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
            
        return messages[-limit:]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics.
        
        Returns:
            Dict: Statistics.
        """
        return {
            "total_messages": len(self._message_history),
            "subscribers": list(self._subscribers.keys()),
            "pending_responses": len(self._pending_responses),
            "message_counts": dict(self._message_counts)
        }


class A2AProtocol:
    """Agent-to-Agent communication protocol.
    
    Provides high-level protocol operations for agent
    communication including request-response patterns,
    task delegation, and feedback loops.
    
    Attributes:
        agent_name: Name of the agent using this protocol.
        message_bus: Shared message bus.
        
    Example:
        >>> protocol = A2AProtocol("orchestrator", bus)
        >>> response = await protocol.request("retriever", "search", {"query": "AI"})
    """
    
    def __init__(
        self,
        agent_name: str,
        message_bus: Optional[MessageBus] = None
    ):
        """Initialize the protocol.
        
        Args:
            agent_name: Name of the agent.
            message_bus: Shared message bus.
        """
        self.agent_name = agent_name
        self.message_bus = message_bus or MessageBus()
        self._handlers: Dict[str, Callable] = {}
        
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register a handler for a task type.
        
        Args:
            task_type: Type of task.
            handler: Handler function.
        """
        self._handlers[task_type] = handler
        
        # Subscribe to message bus
        async def message_handler(message: AgentMessage) -> Optional[AgentMessage]:
            content = message.content
            if isinstance(content, dict) and content.get("task_type") == task_type:
                result = await self._handle_task(message)
                return result
            return None
            
        self.message_bus.subscribe(self.agent_name, message_handler)
        
    async def _handle_task(self, message: AgentMessage) -> AgentMessage:
        """Handle an incoming task message.
        
        Args:
            message: Task message.
            
        Returns:
            AgentMessage: Result message.
        """
        content = message.content
        task_type = content.get("task_type", "")
        parameters = content.get("parameters", {})
        
        handler = self._handlers.get(task_type)
        
        if not handler:
            return message.reply(
                {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                },
                MessageType.ERROR
            )
            
        try:
            start_time = datetime.now()
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(parameters)
            else:
                result = handler(parameters)
                
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return message.reply(
                {
                    "success": True,
                    "data": result,
                    "execution_time_ms": execution_time
                },
                MessageType.RESULT
            )
            
        except Exception as e:
            return message.reply(
                {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                MessageType.ERROR
            )
            
    async def request(
        self,
        to_agent: str,
        task_type: str,
        parameters: Dict[str, Any],
        timeout: float = 30.0,
        priority: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Send a request and wait for response.
        
        Args:
            to_agent: Target agent.
            task_type: Type of task.
            parameters: Task parameters.
            timeout: Response timeout.
            priority: Task priority.
            
        Returns:
            Optional[Dict]: Response data or None.
        """
        message = AgentMessage(
            from_agent=self.agent_name,
            to_agent=to_agent,
            message_type=MessageType.TASK,
            content={
                "task_type": task_type,
                "parameters": parameters
            },
            priority=priority
        )
        
        response = await self.message_bus.send(message, wait_response=True, timeout=timeout)
        
        if response:
            return response.content
        return None
        
    async def notify(
        self,
        to_agent: str,
        message_type: MessageType,
        content: Any
    ) -> None:
        """Send a notification without waiting for response.
        
        Args:
            to_agent: Target agent.
            message_type: Type of notification.
            content: Notification content.
        """
        message = AgentMessage(
            from_agent=self.agent_name,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )
        
        await self.message_bus.send(message, wait_response=False)
        
    async def send_feedback(
        self,
        to_agent: str,
        rating: float,
        suggestions: List[str] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Send feedback to an agent.
        
        Args:
            to_agent: Target agent.
            rating: Quality rating (0-10).
            suggestions: Improvement suggestions.
            correlation_id: Related task ID.
        """
        message = AgentMessage(
            from_agent=self.agent_name,
            to_agent=to_agent,
            message_type=MessageType.FEEDBACK,
            content={
                "rating": rating,
                "suggestions": suggestions or []
            },
            correlation_id=correlation_id or ""
        )
        
        await self.message_bus.send(message, wait_response=False)
        
    async def broadcast_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast status update to all agents.
        
        Args:
            status: Status string.
            details: Additional details.
            
        Returns:
            int: Number of agents notified.
        """
        message = AgentMessage(
            from_agent=self.agent_name,
            to_agent="*",  # Broadcast indicator
            message_type=MessageType.STATUS,
            content={
                "status": status,
                "details": details or {}
            }
        )
        
        return await self.message_bus.broadcast(message, exclude={self.agent_name})
        
    def get_message_history(
        self,
        message_type: Optional[MessageType] = None,
        limit: int = 50
    ) -> List[AgentMessage]:
        """Get message history for this agent.
        
        Args:
            message_type: Filter by type.
            limit: Maximum messages.
            
        Returns:
            List[AgentMessage]: Message history.
        """
        return self.message_bus.get_history(
            agent_name=self.agent_name,
            message_type=message_type,
            limit=limit
        )
