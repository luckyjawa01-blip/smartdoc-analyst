"""Memory Manager for SmartDoc Analyst.

This module provides a unified interface for the three-tier
memory architecture: working, episodic, and semantic memory.
"""

from typing import Any, Dict, List, Optional
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .vector_store import VectorStoreMemory


class MemoryManager:
    """Unified memory management for the three-tier architecture.
    
    Manages three types of memory:
    1. Working Memory (Short-term): Current task context
    2. Episodic Memory (Session): Conversation history
    3. Semantic Memory (Long-term): Persistent knowledge and documents
    
    Attributes:
        working_memory: Short-term memory for current context.
        episodic_memory: Long-term memory for session history.
        semantic_memory: Vector store for document embeddings.
        
    Example:
        >>> manager = MemoryManager()
        >>> manager.add_to_context("User asked about AI trends")
        >>> manager.store_fact("AI adoption grew 50% in 2024")
        >>> context = manager.get_context_for_llm()
    """
    
    def __init__(
        self,
        working_memory_size: int = 100,
        persistence_path: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        embedding_model: Optional[Any] = None
    ):
        """Initialize the memory manager.
        
        Args:
            working_memory_size: Size of working memory buffer.
            persistence_path: Path for long-term memory persistence.
            vector_store_path: Path for vector store persistence.
            embedding_model: Model for generating embeddings.
        """
        # Initialize three memory tiers
        self.working_memory = ShortTermMemory(max_items=working_memory_size)
        self.episodic_memory = LongTermMemory(
            persistence_path=persistence_path,
            auto_save=True
        )
        self.semantic_memory = VectorStoreMemory(
            embedding_model=embedding_model,
            persist_directory=vector_store_path
        )
        
    # ==================== Working Memory Operations ====================
    
    def add_to_context(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """Add content to working memory.
        
        Args:
            content: Content to add.
            metadata: Optional metadata.
            importance: Importance score (0-1).
            
        Returns:
            str: Memory entry ID.
        """
        return self.working_memory.add(content, metadata, importance)
        
    def get_recent_context(self, n: int = 10) -> List[Any]:
        """Get recent working memory entries.
        
        Args:
            n: Number of entries to retrieve.
            
        Returns:
            List: Recent memory entries.
        """
        entries = self.working_memory.get_recent(n)
        return [entry.content for entry in entries]
        
    def clear_context(self) -> int:
        """Clear working memory.
        
        Returns:
            int: Number of entries cleared.
        """
        return self.working_memory.clear()
        
    # ==================== Episodic Memory Operations ====================
    
    def store_episode(
        self,
        key: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store an episode in conversation history.
        
        Args:
            key: Unique key for the episode.
            content: Episode content.
            metadata: Optional metadata.
            
        Returns:
            bool: True if stored successfully.
        """
        return self.episodic_memory.store(
            key=key,
            value=content,
            category="learned",
            metadata=metadata
        )
        
    def recall_episode(self, key: str) -> Optional[Any]:
        """Recall an episode from conversation history.
        
        Args:
            key: Episode key.
            
        Returns:
            Optional[Any]: Episode content or None.
        """
        return self.episodic_memory.retrieve(key, category="learned")
        
    def store_fact(
        self,
        key: str,
        fact: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a fact in long-term memory.
        
        Args:
            key: Unique key for the fact.
            fact: Fact content.
            metadata: Optional metadata.
            
        Returns:
            bool: True if stored successfully.
        """
        return self.episodic_memory.store(
            key=key,
            value=fact,
            category="facts",
            metadata=metadata
        )
        
    def recall_fact(self, key: str) -> Optional[Any]:
        """Recall a fact from long-term memory.
        
        Args:
            key: Fact key.
            
        Returns:
            Optional[Any]: Fact content or None.
        """
        return self.episodic_memory.retrieve(key, category="facts")
        
    def store_preference(
        self,
        key: str,
        value: Any
    ) -> bool:
        """Store a user preference.
        
        Args:
            key: Preference key.
            value: Preference value.
            
        Returns:
            bool: True if stored successfully.
        """
        return self.episodic_memory.store(
            key=key,
            value=value,
            category="preferences"
        )
        
    def get_preference(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a user preference.
        
        Args:
            key: Preference key.
            default: Default value if not found.
            
        Returns:
            Any: Preference value or default.
        """
        value = self.episodic_memory.retrieve(key, category="preferences")
        return value if value is not None else default
        
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search across all memory types.
        
        Args:
            query: Search query.
            
        Returns:
            List[Dict]: Search results from all memory types.
        """
        results = []
        
        # Search episodic memory
        episodic_results = self.episodic_memory.search(query)
        for r in episodic_results:
            r["memory_type"] = "episodic"
            results.append(r)
            
        # Search working memory
        working_results = self.working_memory.search(query)
        for entry in working_results:
            results.append({
                "memory_type": "working",
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat()
            })
            
        return results
        
    # ==================== Semantic Memory Operations ====================
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Add documents to semantic memory.
        
        Args:
            documents: List of documents with content and metadata.
            
        Returns:
            List[str]: Added document IDs.
        """
        return self.semantic_memory.add_documents(documents)
        
    def search_documents(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents in semantic memory.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional metadata filter.
            
        Returns:
            List[Dict]: Search results.
        """
        return self.semantic_memory.search(query, k, filter)
        
    def get_vector_store(self) -> VectorStoreMemory:
        """Get the underlying vector store.
        
        Returns:
            VectorStoreMemory: Vector store instance.
        """
        return self.semantic_memory
        
    # ==================== Context Operations ====================
    
    def get_context_for_llm(
        self,
        working_window: int = 5,
        include_facts: bool = True,
        include_preferences: bool = True
    ) -> Dict[str, Any]:
        """Get combined context for LLM prompt.
        
        Args:
            working_window: Size of working memory window.
            include_facts: Include relevant facts.
            include_preferences: Include user preferences.
            
        Returns:
            Dict: Combined context dictionary.
        """
        context = {
            "working_memory": self.working_memory.get_context_window(working_window),
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        
        if include_facts:
            context["facts"] = self.episodic_memory.get_category("facts")
            
        if include_preferences:
            context["preferences"] = self.episodic_memory.get_category("preferences")
            
        return context
        
    def build_prompt_context(
        self,
        query: str,
        include_documents: bool = True,
        document_k: int = 3
    ) -> str:
        """Build context string for prompt injection.
        
        Args:
            query: Current query.
            include_documents: Include relevant documents.
            document_k: Number of documents to include.
            
        Returns:
            str: Formatted context string.
        """
        parts = []
        
        # Recent context
        recent = self.get_recent_context(5)
        if recent:
            parts.append("## Recent Context")
            for item in recent:
                parts.append(f"- {item}")
            parts.append("")
            
        # Relevant documents
        if include_documents:
            docs = self.search_documents(query, k=document_k)
            if docs:
                parts.append("## Relevant Documents")
                for doc in docs:
                    content = doc.get("content", "")[:500]
                    source = doc.get("metadata", {}).get("source", "Unknown")
                    parts.append(f"[{source}]: {content}")
                parts.append("")
                
        # Relevant facts
        facts = self.episodic_memory.search(query)
        if facts:
            parts.append("## Relevant Knowledge")
            for fact in facts[:5]:
                parts.append(f"- {fact.get('value', fact.get('content', ''))}")
                
        return "\n".join(parts)
        
    # ==================== Management Operations ====================
    
    def clear_all(self) -> Dict[str, int]:
        """Clear all memory stores.
        
        Returns:
            Dict: Count of cleared items per store.
        """
        return {
            "working_memory": self.working_memory.clear(),
            "episodic_memory": self.episodic_memory.clear(),
            "semantic_memory": self.semantic_memory.clear()
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all memory stores.
        
        Returns:
            Dict: Combined statistics.
        """
        return {
            "working_memory": self.working_memory.get_stats(),
            "episodic_memory": self.episodic_memory.get_stats(),
            "semantic_memory": self.semantic_memory.get_stats()
        }
        
    def export_all(self) -> Dict[str, Any]:
        """Export all memory data.
        
        Returns:
            Dict: Complete memory export.
        """
        return {
            "episodic": self.episodic_memory.export(),
            "semantic_stats": self.semantic_memory.get_stats(),
            "working_stats": self.working_memory.get_stats()
        }
        
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MemoryManager("
            f"working={len(self.working_memory)}, "
            f"episodic={len(self.episodic_memory)}, "
            f"semantic={len(self.semantic_memory)})"
        )
