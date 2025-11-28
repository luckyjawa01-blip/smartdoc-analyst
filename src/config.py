"""Configuration management for SmartDoc Analyst.

This module provides centralized configuration using Pydantic Settings,
supporting environment variables, .env files, and sensible defaults.
"""

from functools import lru_cache
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support.
    
    All settings can be overridden via environment variables with the
    SMARTDOC_ prefix (e.g., SMARTDOC_GEMINI_API_KEY).
    
    Attributes:
        gemini_api_key: Google Gemini API key for LLM access.
        model_name: Name of the Gemini model to use.
        embedding_model: Model for text embeddings.
        max_tokens: Maximum tokens per LLM response.
        temperature: LLM temperature for response generation.
        max_retries: Maximum retry attempts for failed operations.
        rate_limit_rpm: Rate limit in requests per minute.
        vector_store_path: Path to persistent vector store.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        enable_tracing: Enable distributed tracing.
        enable_metrics: Enable metrics collection.
    """
    
    # API Configuration
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    model_name: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    
    # LLM Parameters
    max_tokens: int = Field(default=4096, ge=1, le=32768, description="Max tokens per response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    
    # Rate Limiting & Retries
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry attempts")
    rate_limit_rpm: int = Field(default=60, ge=1, description="Requests per minute limit")
    
    # Storage
    vector_store_path: str = Field(default="./chroma_db", description="Vector store path")
    
    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    
    # Safety
    max_input_length: int = Field(default=10000, description="Max input text length")
    enable_safety_guards: bool = Field(default=True, description="Enable safety guards")
    
    # Agent Configuration
    max_agent_iterations: int = Field(default=10, ge=1, le=50, description="Max agent iterations")
    parallel_agents: bool = Field(default=True, description="Enable parallel agent execution")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "SMARTDOC_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.
    
    Returns:
        Settings: Application settings instance.
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.model_name)
        'gemini-1.5-flash'
    """
    return Settings()


# Convenience export
settings = get_settings()
