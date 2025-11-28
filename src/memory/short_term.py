"""Short-term memory for SmartDoc Analyst.

This module provides working memory for the current task context,
maintaining conversation history and recent interactions.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single memory entry.
    
    Attributes:
        content: The memory content.
        timestamp: When the memory was created.
        metadata: Additional information about the memory.
        importance: Importance score (0-1) for retention.
    """
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5


class ShortTermMemory:
    """Working memory for current task context.
    
    Maintains a fixed-size buffer of recent memories using
    a sliding window approach. Older memories are automatically
    pruned to maintain the window size.
    
    Attributes:
        max_items: Maximum number of items to retain.
        buffer: Deque buffer for memory storage.
        
    Example:
        >>> memory = ShortTermMemory(max_items=100)
        >>> memory.add("User asked about AI", metadata={"type": "query"})
        >>> recent = memory.get_recent(5)
    """
    
    def __init__(self, max_items: int = 100):
        """Initialize short-term memory.
        
        Args:
            max_items: Maximum items to retain in memory.
        """
        self.max_items = max_items
        self.buffer: deque = deque(maxlen=max_items)
        self._access_count = 0
        
    def add(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """Add a new memory entry.
        
        Args:
            content: Content to remember.
            metadata: Optional metadata.
            importance: Importance score (0-1).
            
        Returns:
            str: Memory entry ID.
        """
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance=importance
        )
        
        self.buffer.append(entry)
        entry_id = f"stm_{len(self.buffer)}_{entry.timestamp.timestamp()}"
        entry.metadata["id"] = entry_id
        
        return entry_id
        
    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get the most recent n memories.
        
        Args:
            n: Number of recent memories to retrieve.
            
        Returns:
            List[MemoryEntry]: Recent memory entries.
        """
        self._access_count += 1
        entries = list(self.buffer)[-n:]
        return entries
        
    def get_by_metadata(
        self,
        key: str,
        value: Any
    ) -> List[MemoryEntry]:
        """Get memories matching metadata criteria.
        
        Args:
            key: Metadata key to match.
            value: Value to match.
            
        Returns:
            List[MemoryEntry]: Matching memories.
        """
        return [
            entry for entry in self.buffer
            if entry.metadata.get(key) == value
        ]
        
    def search(self, query: str) -> List[MemoryEntry]:
        """Search memories by content.
        
        Args:
            query: Search query string.
            
        Returns:
            List[MemoryEntry]: Matching memories.
        """
        query_lower = query.lower()
        results = []
        
        for entry in self.buffer:
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                results.append(entry)
                
        return results
        
    def clear(self) -> int:
        """Clear all memories.
        
        Returns:
            int: Number of entries cleared.
        """
        count = len(self.buffer)
        self.buffer.clear()
        return count
        
    def get_context_window(
        self,
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Get formatted context window for LLM.
        
        Args:
            window_size: Number of entries to include.
            
        Returns:
            Dict: Formatted context dictionary.
        """
        recent = self.get_recent(window_size)
        
        return {
            "memories": [
                {
                    "content": entry.content,
                    "timestamp": entry.timestamp.isoformat(),
                    "importance": entry.importance
                }
                for entry in recent
            ],
            "total_memories": len(self.buffer),
            "window_size": len(recent)
        }
        
    def prune_by_importance(self, threshold: float = 0.3) -> int:
        """Remove low-importance memories.
        
        Args:
            threshold: Minimum importance to retain.
            
        Returns:
            int: Number of entries removed.
        """
        original_count = len(self.buffer)
        
        # Keep only memories above threshold
        kept = [e for e in self.buffer if e.importance >= threshold]
        
        self.buffer.clear()
        self.buffer.extend(kept)
        
        return original_count - len(self.buffer)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dict: Memory statistics.
        """
        if not self.buffer:
            return {
                "total_entries": 0,
                "access_count": self._access_count,
                "avg_importance": 0.0
            }
            
        return {
            "total_entries": len(self.buffer),
            "max_capacity": self.max_items,
            "utilization": len(self.buffer) / self.max_items,
            "access_count": self._access_count,
            "avg_importance": sum(e.importance for e in self.buffer) / len(self.buffer),
            "oldest_entry": self.buffer[0].timestamp.isoformat() if self.buffer else None,
            "newest_entry": self.buffer[-1].timestamp.isoformat() if self.buffer else None
        }
        
    def __len__(self) -> int:
        """Return number of memory entries."""
        return len(self.buffer)
        
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ShortTermMemory(items={len(self.buffer)}, max={self.max_items})"
