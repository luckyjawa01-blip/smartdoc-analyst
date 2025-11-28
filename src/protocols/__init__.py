"""Protocols module for SmartDoc Analyst.

This module provides communication protocols for agent interactions:
- A2A: Agent-to-Agent protocol for inter-agent communication
- Messages: Standard message definitions and types
"""

from .messages import (
    MessageType,
    AgentMessage,
    TaskMessage,
    ResultMessage,
    ErrorMessage,
    FeedbackMessage
)
from .a2a import A2AProtocol, MessageBus

__all__ = [
    "MessageType",
    "AgentMessage",
    "TaskMessage",
    "ResultMessage",
    "ErrorMessage",
    "FeedbackMessage",
    "A2AProtocol",
    "MessageBus",
]
