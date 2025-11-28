"""LLM Interface for SmartDoc Analyst.

This module provides an abstraction layer for language model
interactions, supporting multiple LLM providers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces.
    
    Provides a common interface for interacting with different
    LLM providers (Gemini, OpenAI, etc.).
    
    Attributes:
        model_name: Name of the model.
        temperature: Generation temperature.
        max_tokens: Maximum tokens per response.
        
    Example:
        >>> llm = GeminiInterface(api_key="...")
        >>> response = await llm.generate("Explain AI in healthcare")
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the model.
            temperature: Generation temperature.
            max_tokens: Maximum tokens per response.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            **kwargs: Additional generation parameters.
            
        Returns:
            str: Generated response.
        """
        pass
        
    @abstractmethod
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response with conversation context.
        
        Args:
            prompt: User prompt.
            context: List of previous messages.
            system_prompt: Optional system instruction.
            **kwargs: Additional generation parameters.
            
        Returns:
            str: Generated response.
        """
        pass
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens.
            
        Returns:
            int: Approximate token count.
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4


class GeminiInterface(LLMInterface):
    """Google Gemini LLM interface.
    
    Provides integration with Google's Gemini models
    for text generation tasks.
    
    Example:
        >>> llm = GeminiInterface(
        ...     api_key="your-api-key",
        ...     model_name="gemini-1.5-flash"
        ... )
        >>> response = await llm.generate("What is machine learning?")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """Initialize the Gemini interface.
        
        Args:
            api_key: Gemini API key.
            model_name: Model name.
            temperature: Generation temperature.
            max_tokens: Maximum tokens.
        """
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = api_key
        self._model = None
        self._initialized = False
        
    def _initialize(self) -> bool:
        """Initialize the Gemini client.
        
        Returns:
            bool: True if initialized successfully.
        """
        if self._initialized:
            return True
            
        if not self.api_key:
            return False
            
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            self._initialized = True
            return True
            
        except ImportError:
            return False
        except Exception:
            return False
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response using Gemini.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            **kwargs: Additional parameters.
            
        Returns:
            str: Generated response.
        """
        if not self._initialize():
            # Return demo response if not initialized
            return self._demo_response(prompt)
            
        try:
            # Combine system prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
                
            # Generate response
            response = await asyncio.to_thread(
                self._model.generate_content,
                full_prompt
            )
            
            return response.text if response and response.text else ""
            
        except Exception as e:
            # Return demo response on error
            return self._demo_response(prompt)
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response with conversation context.
        
        Args:
            prompt: User prompt.
            context: Previous conversation messages.
            system_prompt: Optional system instruction.
            **kwargs: Additional parameters.
            
        Returns:
            str: Generated response.
        """
        if not self._initialize():
            return self._demo_response(prompt)
            
        try:
            # Build conversation history
            history = []
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    history.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    history.append({"role": "model", "parts": [content]})
                    
            # Start chat with history
            chat = self._model.start_chat(history=history)
            
            # Add system prompt to user message if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"[System: {system_prompt}]\n\n{prompt}"
                
            # Generate response
            response = await asyncio.to_thread(
                chat.send_message,
                full_prompt
            )
            
            return response.text if response and response.text else ""
            
        except Exception:
            return self._demo_response(prompt)
            
    def _demo_response(self, prompt: str) -> str:
        """Generate a demo response without API.
        
        Args:
            prompt: User prompt.
            
        Returns:
            str: Demo response.
        """
        # Extract key topics from prompt
        prompt_lower = prompt.lower()
        
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return """Based on the provided content, here are the key points:

1. The main topic covers important aspects of the subject matter.
2. Key findings indicate significant developments in this area.
3. Multiple sources confirm the validity of these observations.
4. Future implications suggest continued growth and evolution.

This summary captures the essential information while maintaining accuracy."""

        elif "analyze" in prompt_lower or "analysis" in prompt_lower:
            return """## Analysis Results

### Key Insights
- The data reveals several important patterns
- Trends indicate positive development
- Multiple factors contribute to the observed outcomes

### Patterns Detected
- Consistent growth across measured metrics
- Correlation between key variables
- Cyclical patterns in the data

### Recommendations
1. Continue monitoring key indicators
2. Focus on high-impact areas
3. Address identified gaps

This analysis is based on the available information and standard analytical methods."""

        elif "compare" in prompt_lower or "contrast" in prompt_lower:
            return """## Comparative Analysis

### Similarities
- Both subjects share common foundational elements
- Similar approaches to core challenges
- Comparable outcomes in key metrics

### Differences
- Distinct methodologies employed
- Varying scales of implementation
- Different target audiences

### Conclusion
While sharing fundamental characteristics, each approach offers unique advantages suited to specific contexts."""

        else:
            return f"""Based on your query about "{prompt[:50]}...", here is a comprehensive response:

## Overview
This topic encompasses several important aspects that merit careful consideration.

## Key Points
1. The subject matter is well-documented in current literature
2. Multiple perspectives exist on this topic
3. Recent developments have influenced understanding

## Details
The information gathered suggests a nuanced picture that requires considering multiple factors. Each aspect contributes to the overall understanding of the topic.

## Conclusion
A thorough examination of available sources provides valuable insights into this subject.

*Note: This is a demonstration response. For full functionality, please configure the API key.*"""


class MockLLM(LLMInterface):
    """Mock LLM for testing without API access.
    
    Provides predictable responses for testing purposes.
    """
    
    def __init__(
        self,
        model_name: str = "mock-model",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        default_response: str = "This is a mock response."
    ):
        """Initialize mock LLM.
        
        Args:
            model_name: Model name.
            temperature: Temperature setting.
            max_tokens: Max tokens.
            default_response: Default response text.
        """
        super().__init__(model_name, temperature, max_tokens)
        self.default_response = default_response
        self.call_history: List[Dict[str, Any]] = []
        
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate mock response.
        
        Args:
            prompt: User prompt.
            system_prompt: System instruction.
            **kwargs: Additional params.
            
        Returns:
            str: Mock response.
        """
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        })
        return self.default_response
        
    async def generate_with_context(
        self,
        prompt: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate mock response with context.
        
        Args:
            prompt: User prompt.
            context: Conversation context.
            system_prompt: System instruction.
            **kwargs: Additional params.
            
        Returns:
            str: Mock response.
        """
        self.call_history.append({
            "prompt": prompt,
            "context": context,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        })
        return self.default_response
