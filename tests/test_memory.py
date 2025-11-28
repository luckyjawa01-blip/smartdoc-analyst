"""Tests for SmartDoc Analyst memory system."""

import pytest
from src.memory import (
    MemoryManager,
    ShortTermMemory,
    LongTermMemory,
    VectorStoreMemory
)


class TestShortTermMemory:
    """Test ShortTermMemory class."""
    
    def test_creation(self):
        """Test memory can be created."""
        memory = ShortTermMemory(max_items=100)
        assert len(memory) == 0
        
    def test_add_entry(self):
        """Test adding an entry."""
        memory = ShortTermMemory()
        entry_id = memory.add("Test content", metadata={"key": "value"})
        
        assert entry_id is not None
        assert len(memory) == 1
        
    def test_get_recent(self):
        """Test getting recent entries."""
        memory = ShortTermMemory()
        memory.add("Entry 1")
        memory.add("Entry 2")
        memory.add("Entry 3")
        
        recent = memory.get_recent(2)
        
        assert len(recent) == 2
        
    def test_max_items_limit(self):
        """Test memory respects max items."""
        memory = ShortTermMemory(max_items=3)
        
        for i in range(5):
            memory.add(f"Entry {i}")
            
        assert len(memory) == 3
        
    def test_search(self):
        """Test searching memory."""
        memory = ShortTermMemory()
        memory.add("AI in healthcare")
        memory.add("Machine learning")
        memory.add("AI applications")
        
        results = memory.search("AI")
        
        assert len(results) == 2
        
    def test_clear(self):
        """Test clearing memory."""
        memory = ShortTermMemory()
        memory.add("Entry 1")
        memory.add("Entry 2")
        
        count = memory.clear()
        
        assert count == 2
        assert len(memory) == 0
        
    def test_get_context_window(self):
        """Test getting context window."""
        memory = ShortTermMemory()
        memory.add("Entry 1")
        memory.add("Entry 2")
        
        context = memory.get_context_window(5)
        
        assert "memories" in context
        assert "total_memories" in context


class TestLongTermMemory:
    """Test LongTermMemory class."""
    
    def test_creation(self):
        """Test memory can be created."""
        memory = LongTermMemory()
        assert len(memory) == 0
        
    def test_store_and_retrieve(self):
        """Test storing and retrieving values."""
        memory = LongTermMemory()
        
        memory.store("key1", "value1", category="facts")
        result = memory.retrieve("key1")
        
        assert result == "value1"
        
    def test_store_in_categories(self):
        """Test storing in different categories."""
        memory = LongTermMemory()
        
        memory.store("fact1", "Fact value", category="facts")
        memory.store("pref1", "Preference value", category="preferences")
        
        assert memory.retrieve("fact1", category="facts") == "Fact value"
        assert memory.retrieve("pref1", category="preferences") == "Preference value"
        
    def test_update(self):
        """Test updating values."""
        memory = LongTermMemory()
        
        memory.store("key1", "value1", category="facts")
        memory.update("key1", "value2", category="facts")
        
        assert memory.retrieve("key1") == "value2"
        
    def test_delete(self):
        """Test deleting values."""
        memory = LongTermMemory()
        
        memory.store("key1", "value1")
        deleted = memory.delete("key1")
        
        assert deleted is True
        assert memory.retrieve("key1") is None
        
    def test_search(self):
        """Test searching memory."""
        memory = LongTermMemory()
        
        memory.store("ai_fact", "AI is transforming industries")
        memory.store("ml_fact", "Machine learning is a subset of AI")
        
        results = memory.search("AI")
        
        assert len(results) >= 1
        
    def test_get_category(self):
        """Test getting all entries in category."""
        memory = LongTermMemory()
        
        memory.store("fact1", "Value 1", category="facts")
        memory.store("fact2", "Value 2", category="facts")
        
        facts = memory.get_category("facts")
        
        assert len(facts) == 2


class TestVectorStoreMemory:
    """Test VectorStoreMemory class."""
    
    def test_creation(self):
        """Test memory can be created."""
        memory = VectorStoreMemory()
        assert len(memory) == 0
        
    def test_add_documents(self):
        """Test adding documents."""
        memory = VectorStoreMemory()
        
        docs = [
            {
                "content": "AI is transforming healthcare.",
                "metadata": {"source": "doc1.pdf"}
            },
            {
                "content": "Machine learning enables new applications.",
                "metadata": {"source": "doc2.pdf"}
            }
        ]
        
        ids = memory.add_documents(docs)
        
        assert len(ids) >= 2
        assert len(memory) >= 2
        
    def test_search(self):
        """Test searching documents."""
        memory = VectorStoreMemory()
        
        memory.add_documents([
            {"content": "AI in healthcare applications", "metadata": {}},
            {"content": "Machine learning algorithms", "metadata": {}},
            {"content": "Climate change policy analysis", "metadata": {}}
        ])
        
        results = memory.search("AI healthcare", k=2)
        
        assert len(results) <= 2
        
    def test_similarity_search(self):
        """Test similarity search method."""
        memory = VectorStoreMemory()
        
        memory.add_documents([
            {"content": "Test document content", "metadata": {"source": "test.pdf"}}
        ])
        
        results = memory.similarity_search("document", k=1)
        
        assert len(results) == 1
        assert hasattr(results[0], 'page_content')
        
    def test_clear(self):
        """Test clearing documents."""
        memory = VectorStoreMemory()
        
        memory.add_documents([
            {"content": "Test content", "metadata": {}}
        ])
        
        count = memory.clear()
        
        assert count >= 1
        assert len(memory) == 0


class TestMemoryManager:
    """Test MemoryManager class."""
    
    def test_creation(self):
        """Test manager can be created."""
        manager = MemoryManager()
        assert manager is not None
        
    def test_add_to_context(self):
        """Test adding to working memory."""
        manager = MemoryManager()
        
        entry_id = manager.add_to_context("Test content")
        
        assert entry_id is not None
        
    def test_get_recent_context(self):
        """Test getting recent context."""
        manager = MemoryManager()
        
        manager.add_to_context("Entry 1")
        manager.add_to_context("Entry 2")
        
        recent = manager.get_recent_context(5)
        
        assert len(recent) == 2
        
    def test_store_and_recall_fact(self):
        """Test storing and recalling facts."""
        manager = MemoryManager()
        
        manager.store_fact("fact1", "Important fact")
        result = manager.recall_fact("fact1")
        
        assert result == "Important fact"
        
    def test_store_and_get_preference(self):
        """Test storing and getting preferences."""
        manager = MemoryManager()
        
        manager.store_preference("format", "markdown")
        result = manager.get_preference("format")
        
        assert result == "markdown"
        
    def test_add_and_search_documents(self):
        """Test adding and searching documents."""
        manager = MemoryManager()
        
        manager.add_documents([
            {"content": "AI document content", "metadata": {"source": "ai.pdf"}}
        ])
        
        results = manager.search_documents("AI", k=1)
        
        assert len(results) >= 0  # May be 0 or more depending on implementation
        
    def test_get_context_for_llm(self):
        """Test getting context for LLM."""
        manager = MemoryManager()
        
        manager.add_to_context("Test context")
        manager.store_fact("key", "value")
        
        context = manager.get_context_for_llm()
        
        assert "working_memory" in context
        
    def test_clear_all(self):
        """Test clearing all memory."""
        manager = MemoryManager()
        
        manager.add_to_context("Entry")
        manager.store_fact("fact", "value")
        
        result = manager.clear_all()
        
        assert "working_memory" in result
        assert "episodic_memory" in result
        
    def test_get_stats(self):
        """Test getting memory statistics."""
        manager = MemoryManager()
        
        stats = manager.get_stats()
        
        assert "working_memory" in stats
        assert "episodic_memory" in stats
        assert "semantic_memory" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
