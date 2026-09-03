"""
Agents module for LangGraph node logic.

This module contains agent implementations that serve as nodes
in the LangGraph workflow. Each agent performs a specific task
in the knowledge discovery pipeline.
"""

from src.agents.predictor import (
    predict_llm,
    predict_llm_safe,
    predict_llm_stage4,
    predict_llm_stage4_safe,
)


def __getattr__(name):
    if name in {"DiscoveryAgent", "discover_and_validate"}:
        from src.agents.discovery_agent import DiscoveryAgent, discover_and_validate

        return {"DiscoveryAgent": DiscoveryAgent, "discover_and_validate": discover_and_validate}[name]
    if name == "extract_interactions":
        from src.agents.extractor import extract_interactions

        return extract_interactions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DiscoveryAgent',
    'discover_and_validate',
    'extract_interactions',
    'predict_llm',
    'predict_llm_safe',
    'predict_llm_stage4',
    'predict_llm_stage4_safe',
]
