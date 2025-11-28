"""Memory module for SmartDoc Analyst.

This module provides a three-tier memory architecture:
- ShortTermMemory: Working memory for current task context
- LongTermMemory: Persistent knowledge storage
- VectorStore: Document embeddings for semantic search
- MemoryManager: Unified interface for all memory operations
"""

from .memory_manager import MemoryManager
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .vector_store import VectorStoreMemory

__all__ = [
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
    "VectorStoreMemory",
]
