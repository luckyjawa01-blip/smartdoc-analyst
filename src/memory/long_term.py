"""Long-term memory for SmartDoc Analyst.

This module provides persistent knowledge storage
for facts, learned information, and user preferences.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class LongTermMemory:
    """Persistent knowledge storage.
    
    Stores facts, learned information, and user preferences
    with optional persistence to disk for cross-session memory.
    
    Attributes:
        storage: In-memory storage dictionary.
        persistence_path: Optional path for disk persistence.
        
    Example:
        >>> memory = LongTermMemory(persistence_path="./memory.json")
        >>> memory.store("user_pref_format", "markdown", category="preferences")
        >>> value = memory.retrieve("user_pref_format")
    """
    
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """Initialize long-term memory.
        
        Args:
            persistence_path: Path to persistence file.
            auto_save: Automatically save on changes.
        """
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.auto_save = auto_save
        self.storage: Dict[str, Dict[str, Any]] = {
            "facts": {},
            "preferences": {},
            "learned": {},
            "entities": {}
        }
        
        # Load existing data if available
        if self.persistence_path and self.persistence_path.exists():
            self._load()
            
    def store(
        self,
        key: str,
        value: Any,
        category: str = "facts",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a value in long-term memory.
        
        Args:
            key: Unique key for the value.
            value: Value to store.
            category: Storage category.
            metadata: Optional metadata.
            
        Returns:
            bool: True if stored successfully.
        """
        if category not in self.storage:
            self.storage[category] = {}
            
        self.storage[category][key] = {
            "value": value,
            "stored_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "access_count": 0,
            "metadata": metadata or {}
        }
        
        if self.auto_save:
            self._save()
            
        return True
        
    def retrieve(
        self,
        key: str,
        category: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve a value from long-term memory.
        
        Args:
            key: Key to retrieve.
            category: Optional category to search in.
            
        Returns:
            Optional[Any]: Retrieved value or None.
        """
        if category:
            if category in self.storage and key in self.storage[category]:
                entry = self.storage[category][key]
                entry["access_count"] += 1
                entry["last_accessed"] = datetime.now().isoformat()
                return entry["value"]
            return None
            
        # Search all categories
        for cat_storage in self.storage.values():
            if key in cat_storage:
                entry = cat_storage[key]
                entry["access_count"] += 1
                entry["last_accessed"] = datetime.now().isoformat()
                return entry["value"]
                
        return None
        
    def update(
        self,
        key: str,
        value: Any,
        category: str = "facts"
    ) -> bool:
        """Update an existing memory entry.
        
        Args:
            key: Key to update.
            value: New value.
            category: Storage category.
            
        Returns:
            bool: True if updated successfully.
        """
        if category in self.storage and key in self.storage[category]:
            self.storage[category][key]["value"] = value
            self.storage[category][key]["updated_at"] = datetime.now().isoformat()
            
            if self.auto_save:
                self._save()
            return True
            
        return False
        
    def delete(self, key: str, category: Optional[str] = None) -> bool:
        """Delete a memory entry.
        
        Args:
            key: Key to delete.
            category: Optional category.
            
        Returns:
            bool: True if deleted.
        """
        if category:
            if category in self.storage and key in self.storage[category]:
                del self.storage[category][key]
                if self.auto_save:
                    self._save()
                return True
            return False
            
        # Search all categories
        for cat_storage in self.storage.values():
            if key in cat_storage:
                del cat_storage[key]
                if self.auto_save:
                    self._save()
                return True
                
        return False
        
    def search(
        self,
        query: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search memory entries by query.
        
        Args:
            query: Search query.
            category: Optional category to search.
            
        Returns:
            List[Dict]: Matching entries.
        """
        results = []
        query_lower = query.lower()
        
        categories = [category] if category else self.storage.keys()
        
        for cat in categories:
            if cat not in self.storage:
                continue
                
            for key, entry in self.storage[cat].items():
                # Search in key
                if query_lower in key.lower():
                    results.append({
                        "key": key,
                        "category": cat,
                        "value": entry["value"],
                        "metadata": entry.get("metadata", {})
                    })
                    continue
                    
                # Search in value
                value_str = str(entry["value"]).lower()
                if query_lower in value_str:
                    results.append({
                        "key": key,
                        "category": cat,
                        "value": entry["value"],
                        "metadata": entry.get("metadata", {})
                    })
                    
        return results
        
    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all entries in a category.
        
        Args:
            category: Category name.
            
        Returns:
            Dict: All entries in the category.
        """
        if category in self.storage:
            return {
                key: entry["value"]
                for key, entry in self.storage[category].items()
            }
        return {}
        
    def get_recent(
        self,
        n: int = 10,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get most recently accessed entries.
        
        Args:
            n: Number of entries to return.
            category: Optional category filter.
            
        Returns:
            List[Dict]: Recent entries.
        """
        all_entries = []
        
        categories = [category] if category else self.storage.keys()
        
        for cat in categories:
            if cat not in self.storage:
                continue
                
            for key, entry in self.storage[cat].items():
                all_entries.append({
                    "key": key,
                    "category": cat,
                    "value": entry["value"],
                    "last_accessed": entry.get("last_accessed", entry.get("stored_at")),
                    "access_count": entry.get("access_count", 0)
                })
                
        # Sort by last accessed
        all_entries.sort(
            key=lambda x: x.get("last_accessed", ""),
            reverse=True
        )
        
        return all_entries[:n]
        
    def clear(self, category: Optional[str] = None) -> int:
        """Clear memory entries.
        
        Args:
            category: Optional category to clear.
            
        Returns:
            int: Number of entries cleared.
        """
        if category:
            if category in self.storage:
                count = len(self.storage[category])
                self.storage[category] = {}
                if self.auto_save:
                    self._save()
                return count
            return 0
            
        # Clear all
        count = sum(len(cat) for cat in self.storage.values())
        self.storage = {
            "facts": {},
            "preferences": {},
            "learned": {},
            "entities": {}
        }
        
        if self.auto_save:
            self._save()
            
        return count
        
    def _save(self) -> bool:
        """Save memory to disk.
        
        Returns:
            bool: True if saved successfully.
        """
        if not self.persistence_path:
            return False
            
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self.storage, f, indent=2, default=str)
            return True
        except Exception:
            return False
            
    def _load(self) -> bool:
        """Load memory from disk.
        
        Returns:
            bool: True if loaded successfully.
        """
        if not self.persistence_path or not self.persistence_path.exists():
            return False
            
        try:
            with open(self.persistence_path, 'r') as f:
                loaded = json.load(f)
                self.storage.update(loaded)
            return True
        except Exception:
            return False
            
    def export(self) -> Dict[str, Any]:
        """Export all memory data.
        
        Returns:
            Dict: Complete memory export.
        """
        return {
            "storage": self.storage,
            "exported_at": datetime.now().isoformat(),
            "stats": self.get_stats()
        }
        
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Import memory data.
        
        Args:
            data: Data to import.
            
        Returns:
            bool: True if imported successfully.
        """
        try:
            if "storage" in data:
                self.storage.update(data["storage"])
            else:
                self.storage.update(data)
                
            if self.auto_save:
                self._save()
            return True
        except Exception:
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dict: Memory statistics.
        """
        stats = {
            "total_entries": 0,
            "categories": {}
        }
        
        for category, entries in self.storage.items():
            count = len(entries)
            stats["total_entries"] += count
            stats["categories"][category] = {
                "count": count,
                "total_accesses": sum(
                    e.get("access_count", 0) for e in entries.values()
                )
            }
            
        stats["persistence_enabled"] = self.persistence_path is not None
        
        return stats
        
    def __len__(self) -> int:
        """Return total number of entries."""
        return sum(len(cat) for cat in self.storage.values())
        
    def __repr__(self) -> str:
        """Return string representation."""
        return f"LongTermMemory(entries={len(self)}, persistent={self.persistence_path is not None})"
