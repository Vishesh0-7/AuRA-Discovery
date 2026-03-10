"""
Factory module for pre-configured AI model instances.

This module provides factory functions to create properly configured
instances of language models and other AI services used throughout
the application.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def get_gemini_llm(
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_output_tokens: Optional[int] = 8192,
    top_p: float = 0.95,
    top_k: int = 40,
) -> ChatGoogleGenerativeAI:
    """
    Create a pre-configured ChatGoogleGenerativeAI instance.
    
    Args:
        model_name: The Gemini model to use (default: gemini-2.5-flash)
        temperature: Sampling temperature (0.0 to 1.0)
        max_output_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        Configured ChatGoogleGenerativeAI instance
    
    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
        convert_system_message_to_human=True,  # For compatibility
    )


def get_structured_llm(
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> ChatGoogleGenerativeAI:
    """
    Create a Gemini instance optimized for structured output extraction.
    
    Uses lower temperature for more deterministic outputs, ideal for
    entity extraction and relationship identification.
    
    Args:
        model_name: The Gemini model to use
        temperature: Lower temperature for more focused outputs
    
    Returns:
        Configured ChatGoogleGenerativeAI instance for structured output
    """
    return get_gemini_llm(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=8192,
    )


def get_reasoning_llm(
    model_name: str = "gemini-2.5-flash", 
    temperature: float = 0.9,
) -> ChatGoogleGenerativeAI:
    """
    Create a Gemini instance optimized for creative reasoning and synthesis.
    
    Uses higher temperature for more creative and diverse outputs, ideal for
    hypothesis generation and research synthesis.
    
    Args:
        model_name: The Gemini model to use
        temperature: Higher temperature for more creative outputs
    
    Returns:
        Configured ChatGoogleGenerativeAI instance for reasoning
    """
    return get_gemini_llm(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=8192,
    )


# Pre-configured default instances for convenience
# These can be imported directly: from src.factory import default_llm
default_llm = get_gemini_llm()
structured_llm = get_structured_llm()
reasoning_llm = get_reasoning_llm()
