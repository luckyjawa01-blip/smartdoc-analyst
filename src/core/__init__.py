"""Core module for SmartDoc Analyst.

This module provides the main system components:
- SmartDocAnalyst: Main system class
- LLMInterface: Language model abstraction
- SafetyGuard: Input validation and safety checks
"""

from .system import SmartDocAnalyst
from .llm_interface import LLMInterface, GeminiInterface
from .safety import SafetyGuard, ValidationResult

__all__ = [
    "SmartDocAnalyst",
    "LLMInterface",
    "GeminiInterface",
    "SafetyGuard",
    "ValidationResult",
]
