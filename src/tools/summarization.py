"""Summarization Tool for SmartDoc Analyst.

This tool generates concise summaries of text content
using extractive and abstractive techniques.
"""

from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult


class SummarizationTool(BaseTool):
    """Text summarization tool for generating concise summaries.
    
    Supports both extractive (selecting key sentences) and
    abstractive (rewriting) summarization approaches.
    
    Attributes:
        llm: Language model for abstractive summarization.
        max_summary_length: Maximum summary length in words.
        method: Default summarization method.
        
    Example:
        >>> tool = SummarizationTool(llm=my_llm)
        >>> result = await tool.execute(
        ...     text="Long document text...",
        ...     max_length=100,
        ...     method="abstractive"
        ... )
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        max_summary_length: int = 200,
        method: str = "extractive"
    ):
        """Initialize the summarization tool.
        
        Args:
            llm: Language model for abstractive summarization.
            max_summary_length: Maximum summary length in words.
            method: Default method (extractive/abstractive).
        """
        super().__init__(
            name="summarization",
            description="Generate concise summaries of text content"
        )
        self.llm = llm
        self.max_summary_length = max_summary_length
        self.method = method
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Generate a summary of the input text.
        
        Args:
            text: Text to summarize.
            max_length: Maximum summary length in words.
            method: Summarization method (extractive/abstractive).
            
        Returns:
            ToolResult: Summary result.
        """
        text = kwargs.get("text", "")
        max_length = kwargs.get("max_length", self.max_summary_length)
        method = kwargs.get("method", self.method)
        
        if not text:
            return ToolResult(
                success=False,
                error="Text is required for summarization"
            )
            
        try:
            if method == "abstractive" and self.llm:
                summary = await self._abstractive_summarize(text, max_length)
            else:
                summary = self._extractive_summarize(text, max_length)
                
            return ToolResult(
                success=True,
                data={
                    "summary": summary,
                    "method": method,
                    "original_length": len(text.split()),
                    "summary_length": len(summary.split()),
                    "compression_ratio": len(summary.split()) / max(len(text.split()), 1)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Summarization failed: {str(e)}"
            )
            
    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """Generate extractive summary by selecting key sentences.
        
        Args:
            text: Text to summarize.
            max_length: Maximum summary length in words.
            
        Returns:
            str: Extractive summary.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return text[:max_length * 5] if text else ""
            
        # Score sentences based on position and keyword frequency
        scored = self._score_sentences(sentences, text)
        
        # Select top sentences until max length
        scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        word_count = 0
        
        for sentence, score in scored:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= max_length:
                selected.append((sentence, sentences.index(sentence)))
                word_count += sentence_words
            if word_count >= max_length:
                break
                
        # Sort by original position
        selected.sort(key=lambda x: x[1])
        
        return " ".join(s[0] for s in selected)
        
    async def _abstractive_summarize(self, text: str, max_length: int) -> str:
        """Generate abstractive summary using LLM.
        
        Args:
            text: Text to summarize.
            max_length: Maximum summary length in words.
            
        Returns:
            str: Abstractive summary.
        """
        if not self.llm:
            return self._extractive_summarize(text, max_length)
            
        prompt = f"""Summarize the following text in approximately {max_length} words.
Focus on the key points and main ideas. Be concise and clear.

Text:
{text[:8000]}

Summary:"""
        
        try:
            response = await self.llm.generate(prompt)
            if response:
                # Truncate if needed
                words = response.split()
                if len(words) > max_length * 1.2:
                    response = " ".join(words[:max_length]) + "..."
                return response
        except Exception:
            pass
            
        # Fallback to extractive
        return self._extractive_summarize(text, max_length)
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split.
            
        Returns:
            List[str]: List of sentences.
        """
        # Simple sentence splitting
        import re
        
        # Handle common abbreviations
        text = text.replace("Mr.", "Mr").replace("Mrs.", "Mrs")
        text = text.replace("Dr.", "Dr").replace("etc.", "etc")
        text = text.replace("e.g.", "eg").replace("i.e.", "ie")
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
        
        return sentences
        
    def _score_sentences(
        self,
        sentences: List[str],
        full_text: str
    ) -> List[tuple]:
        """Score sentences for importance.
        
        Args:
            sentences: List of sentences.
            full_text: Complete text for context.
            
        Returns:
            List[tuple]: (sentence, score) pairs.
        """
        # Calculate word frequency
        words = full_text.lower().split()
        word_freq = {}
        for word in words:
            # Skip stop words
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        scored = []
        num_sentences = len(sentences)
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Position score (first and last sentences weighted higher)
            if i == 0:
                score += 2.0
            elif i == num_sentences - 1:
                score += 1.5
            elif i < num_sentences * 0.2:
                score += 1.0
                
            # Word frequency score
            sentence_words = sentence.lower().split()
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word] / max(len(words), 1)
                    
            # Length penalty (prefer medium length sentences)
            length = len(sentence_words)
            if 10 <= length <= 30:
                score += 0.5
            elif length > 50:
                score -= 0.5
                
            # Keyword bonus
            keywords = ['important', 'significant', 'key', 'main', 'conclude',
                       'result', 'finding', 'show', 'demonstrate', 'indicate']
            for kw in keywords:
                if kw in sentence.lower():
                    score += 0.3
                    
            scored.append((sentence, score))
            
        return scored
        
    async def summarize_documents(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 300
    ) -> Dict[str, Any]:
        """Summarize multiple documents.
        
        Args:
            documents: List of document dictionaries.
            max_length: Maximum total summary length.
            
        Returns:
            Dict: Combined summary.
        """
        # Summarize each document
        summaries = []
        per_doc_length = max(50, max_length // max(len(documents), 1))
        
        for doc in documents:
            content = doc.get("content", "")
            if content:
                result = await self.execute(
                    text=content,
                    max_length=per_doc_length
                )
                if result.success:
                    summaries.append({
                        "source": doc.get("metadata", {}).get("source", "Unknown"),
                        "summary": result.data["summary"]
                    })
                    
        # Combine summaries
        combined = " ".join(s["summary"] for s in summaries)
        
        # Final summary of summaries if too long
        if len(combined.split()) > max_length:
            final_result = await self.execute(
                text=combined,
                max_length=max_length
            )
            combined = final_result.data["summary"] if final_result.success else combined
            
        return {
            "combined_summary": combined,
            "document_summaries": summaries,
            "documents_processed": len(summaries)
        }
        
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: Parameter schema.
        """
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to summarize"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum summary length in words",
                    "default": self.max_summary_length
                },
                "method": {
                    "type": "string",
                    "description": "Summarization method",
                    "enum": ["extractive", "abstractive"],
                    "default": self.method
                }
            },
            "required": ["text"]
        }
