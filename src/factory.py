"""
Factory module for pre-configured AI model instances.

This module provides factory functions to create properly configured
instances of language models and other AI services used throughout
the application.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()


def get_ollama_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: Optional[int] = 8192,
    top_p: float = 0.95,
    top_k: int = 40,
) -> ChatOllama:
    """
    Create a pre-configured ChatOllama instance.
    
    Args:
        model_name: The Ollama model to use (defaults to OLLAMA_MODEL or llama3.1:8b)
        temperature: Sampling temperature (0.0 to 1.0)
        max_output_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Configured ChatOllama instance
    
    Notes:
        OLLAMA_BASE_URL can be used to override the Ollama endpoint.
        Defaults to http://localhost:11434.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    resolved_model = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    return ChatOllama(
        model=resolved_model,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
    )


def get_gemini_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: Optional[int] = 8192,
    top_p: float = 0.95,
    top_k: int = 40,
) -> ChatOllama:
    """Backward-compatible wrapper retained for older imports."""
    return get_ollama_llm(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
    )


def get_structured_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.3,
) -> ChatOllama:
    """
    Create an Ollama instance optimized for structured output extraction.
    
    Uses lower temperature for more deterministic outputs, ideal for
    entity extraction and relationship identification.
    
    Args:
        model_name: The Ollama model to use
        temperature: Lower temperature for more focused outputs
    
    Returns:
        Configured ChatOllama instance for structured output
    """
    return get_ollama_llm(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=8192,
    )


def get_reasoning_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.9,
) -> ChatOllama:
    """
    Create an Ollama instance optimized for creative reasoning and synthesis.
    
    Uses higher temperature for more creative and diverse outputs, ideal for
    hypothesis generation and research synthesis.
    
    Args:
        model_name: The Ollama model to use
        temperature: Higher temperature for more creative outputs
    
    Returns:
        Configured ChatOllama instance for reasoning
    """
    return get_ollama_llm(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=8192,
    )


# Pre-configured default instances for convenience
# These can be imported directly: from src.factory import default_llm
default_llm = get_ollama_llm()
structured_llm = get_structured_llm()
reasoning_llm = get_reasoning_llm()
