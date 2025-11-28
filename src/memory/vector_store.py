"""Vector store memory for SmartDoc Analyst.

This module provides document embedding storage and
semantic search capabilities using vector databases.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class DocumentChunk:
    """A chunk of document with embedding.
    
    Attributes:
        id: Unique chunk identifier.
        content: Text content of the chunk.
        embedding: Vector embedding.
        metadata: Document metadata.
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStoreMemory:
    """Vector store for document embeddings and semantic search.
    
    Provides storage for document chunks with their embeddings
    and supports semantic similarity search.
    
    Attributes:
        embedding_model: Model for generating embeddings.
        store: Underlying vector store (ChromaDB, etc.).
        collection_name: Name of the collection.
        
    Example:
        >>> memory = VectorStoreMemory(embedding_model=my_model)
        >>> memory.add_documents([{"content": "AI is transforming...", "metadata": {...}}])
        >>> results = memory.search("artificial intelligence applications")
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        store: Optional[Any] = None,
        collection_name: str = "smartdoc_documents",
        persist_directory: Optional[str] = None
    ):
        """Initialize vector store memory.
        
        Args:
            embedding_model: Model for generating embeddings.
            store: Pre-configured vector store.
            collection_name: Collection name.
            persist_directory: Directory for persistence.
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._store = store
        self._documents: Dict[str, DocumentChunk] = {}
        self._initialized = False
        
        if store is None:
            self._init_store()
            
    def _init_store(self) -> None:
        """Initialize the vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            if self.persist_directory:
                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory
                ))
            else:
                self._client = chromadb.Client()
                
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            self._initialized = True
            
        except ImportError:
            # Use simple in-memory store as fallback
            self._initialized = False
            
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents with content and metadata.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            
        Returns:
            List[str]: IDs of added documents.
        """
        added_ids = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Split into chunks
            chunks = self._split_text(content, chunk_size, chunk_overlap)
            
            for i, chunk_text in enumerate(chunks):
                # Generate ID
                chunk_id = self._generate_id(chunk_text, metadata, i)
                
                # Generate embedding
                embedding = self._get_embedding(chunk_text)
                
                # Create chunk
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    embedding=embedding,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                
                # Store in memory
                self._documents[chunk_id] = chunk
                
                # Store in vector database if available
                if self._initialized and hasattr(self, '_collection'):
                    try:
                        self._collection.add(
                            documents=[chunk_text],
                            metadatas=[chunk.metadata],
                            ids=[chunk_id],
                            embeddings=[embedding] if embedding else None
                        )
                    except Exception:
                        pass
                        
                added_ids.append(chunk_id)
                
        return added_ids
        
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional metadata filter.
            
        Returns:
            List[Dict]: Search results with content and metadata.
        """
        # Try vector store first
        if self._initialized and hasattr(self, '_collection'):
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=filter
                )
                
                documents = []
                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        documents.append({
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                            "score": results["distances"][0][i] if results.get("distances") else None
                        })
                return documents
                
            except Exception:
                pass
                
        # Fallback to simple text search
        return self._simple_search(query, k, filter)
        
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Alias for search method compatible with LangChain interface.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional metadata filter.
            
        Returns:
            List: Search results as document-like objects.
        """
        results = self.search(query, k, filter)
        
        # Return as simple objects with page_content and metadata
        class SimpleDoc:
            def __init__(self, content, metadata, score=None):
                self.page_content = content
                self.metadata = metadata
                self.score = score
                
        return [
            SimpleDoc(r["content"], r.get("metadata", {}), r.get("score"))
            for r in results
        ]
        
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Search with relevance scores.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional filter.
            
        Returns:
            List[tuple]: (document, score) pairs.
        """
        results = self.similarity_search(query, k, filter)
        return [(doc, doc.score or 0.0) for doc in results]
        
    def _simple_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Simple text-based search fallback.
        
        Args:
            query: Search query.
            k: Number of results.
            filter: Optional filter.
            
        Returns:
            List[Dict]: Search results.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_results = []
        
        for doc_id, chunk in self._documents.items():
            # Apply filter if provided
            if filter:
                match = all(
                    chunk.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue
                    
            # Score by word overlap
            content_lower = chunk.content.lower()
            content_words = set(content_lower.split())
            
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)
            
            # Boost if query appears as substring
            if query_lower in content_lower:
                score += 0.5
                
            scored_results.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": score
            })
            
        # Sort by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_results[:k]
        
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split.
            chunk_size: Maximum chunk size.
            overlap: Overlap between chunks.
            
        Returns:
            List[str]: Text chunks.
        """
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for i in range(min(100, end - start)):
                    check_pos = end - i
                    if text[check_pos] in '.!?\n':
                        end = check_pos + 1
                        break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - overlap
            
        return chunks
        
    def _generate_id(
        self,
        content: str,
        metadata: Dict[str, Any],
        index: int
    ) -> str:
        """Generate unique ID for a chunk.
        
        Args:
            content: Chunk content.
            metadata: Chunk metadata.
            index: Chunk index.
            
        Returns:
            str: Unique identifier.
        """
        source = metadata.get("source", "unknown")
        hash_input = f"{source}_{index}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
        
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Optional[List[float]]: Embedding vector.
        """
        if not self.embedding_model:
            return None
            
        try:
            if hasattr(self.embedding_model, 'encode'):
                # Sentence transformers style
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            elif hasattr(self.embedding_model, 'embed_query'):
                # LangChain style
                return self.embedding_model.embed_query(text)
            else:
                return None
        except Exception:
            return None
            
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete documents from the store.
        
        Args:
            ids: Specific IDs to delete.
            filter: Metadata filter for deletion.
            
        Returns:
            int: Number of documents deleted.
        """
        deleted = 0
        
        if ids:
            for doc_id in ids:
                if doc_id in self._documents:
                    del self._documents[doc_id]
                    deleted += 1
                    
            if self._initialized and hasattr(self, '_collection'):
                try:
                    self._collection.delete(ids=ids)
                except Exception:
                    pass
                    
        elif filter:
            # Delete by filter
            to_delete = []
            for doc_id, chunk in self._documents.items():
                match = all(
                    chunk.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if match:
                    to_delete.append(doc_id)
                    
            for doc_id in to_delete:
                del self._documents[doc_id]
                deleted += 1
                
        return deleted
        
    def clear(self) -> int:
        """Clear all documents.
        
        Returns:
            int: Number of documents cleared.
        """
        count = len(self._documents)
        self._documents.clear()
        
        if self._initialized and hasattr(self, '_collection'):
            try:
                # Recreate collection
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(self.collection_name)
            except Exception:
                pass
                
        return count
        
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add text documents (LangChain compatibility).
        
        Args:
            texts: List of text contents.
            metadatas: Optional list of metadata dicts.
            
        Returns:
            List[str]: Added document IDs.
        """
        documents = []
        for i, text in enumerate(texts):
            doc = {
                "content": text,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {}
            }
            documents.append(doc)
            
        return self.add_documents(documents)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dict: Store statistics.
        """
        return {
            "total_documents": len(self._documents),
            "collection_name": self.collection_name,
            "vector_db_initialized": self._initialized,
            "persist_directory": self.persist_directory,
            "embedding_model": str(type(self.embedding_model).__name__) if self.embedding_model else None
        }
        
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self._documents)
        
    def __repr__(self) -> str:
        """Return string representation."""
        return f"VectorStoreMemory(documents={len(self._documents)}, collection='{self.collection_name}')"
