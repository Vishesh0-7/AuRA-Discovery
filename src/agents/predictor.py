"""LLM-based DDI prediction agent.

This module provides a single convenience function, predict_llm,
for classifying a drug pair and abstract as DDI or No DDI with a
confidence score.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.factory import get_structured_llm
from src.schema import DDIPrediction, Stage4LLMPrediction


logger = logging.getLogger(__name__)


PREDICTION_PROMPT = """You are a careful biomedical NLP system for drug-drug interaction (DDI) classification.

Task:
Given a drug pair and a scientific abstract, decide whether the abstract provides evidence of a drug-drug interaction for that pair.

Decision rules:
1. Return "DDI" only if the abstract explicitly indicates an interaction, clinically relevant effect, interaction risk, contraindication, altered exposure, toxicity, potentiation, inhibition, synergy, antagonism, or otherwise a direct interaction involving the given drug pair.
2. Return "No DDI" if the abstract does not clearly support an interaction for the given pair.
3. Use only the provided abstract and the named pair. Do not infer beyond the text.
4. Confidence must reflect how clearly the abstract supports the prediction.
5. Include a short rationale and, if available, the specific evidence text from the abstract.

Drug 1: {drug1}
Drug 2: {drug2}

Abstract:
{text}

Return a concise binary prediction for this specific pair."""


STAGE4_PREDICTION_PROMPT = """You are a strict biomedical DDI classifier.

Given Drug 1, Drug 2, and an abstract, determine whether the abstract supports an interaction for THIS specific pair.

Return ONLY a structured object with exactly these fields:
- interaction: boolean
- confidence_score: number between 0 and 1
- evidence_snippet: short quoted or paraphrased snippet, or null if not present

Rules:
1. interaction=true only if the abstract clearly supports a direct interaction, altered exposure/effect, contraindication, toxicity risk, or clinically relevant interaction for the provided pair.
2. interaction=false if evidence is weak, indirect, unrelated, or absent for the pair.
3. Do not add any fields.
4. Confidence must match evidence clarity.

Drug 1: {drug1}
Drug 2: {drug2}
Abstract: {text}
"""


def predict_llm(
    drug1: str,
    drug2: str,
    text: str,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
) -> DDIPrediction:
    """Predict whether the given drug pair has a DDI in the abstract.

    Args:
        drug1: First drug name.
        drug2: Second drug name.
        text: Abstract or evidence text.
        model_name: Ollama model to use.
        temperature: Sampling temperature; keep low for classification.

    Returns:
        DDIPrediction containing label and confidence.
    """
    if not drug1 or not drug1.strip():
        raise ValueError("drug1 is required")
    if not drug2 or not drug2.strip():
        raise ValueError("drug2 is required")
    if not text or not text.strip():
        raise ValueError("text is required")

    logger.info("Predicting DDI for %s vs %s", drug1, drug2)

    llm = get_structured_llm(model_name=model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(DDIPrediction)

    prompt = PREDICTION_PROMPT.format(drug1=drug1.strip(), drug2=drug2.strip(), text=text.strip())

    result = structured_llm.invoke(prompt)
    logger.info("Prediction complete: %s (confidence=%.3f)", result.label, result.confidence)
    return result


def predict_llm_safe(
    drug1: str,
    drug2: str,
    text: str,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
) -> DDIPrediction:
    """Safe wrapper around predict_llm that returns a fallback negative prediction on failure."""
    try:
        return predict_llm(
            drug1=drug1,
            drug2=drug2,
            text=text,
            model_name=model_name,
            temperature=temperature,
        )
    except Exception as exc:
        logger.error("LLM prediction failed: %s", exc, exc_info=True)
        return DDIPrediction(label="No DDI", confidence=0.0, rationale=str(exc))


def predict_llm_stage4(
    drug1: str,
    drug2: str,
    text: str,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
) -> Stage4LLMPrediction:
    """Strict Stage 4 prediction with schema: interaction/confidence_score/evidence_snippet."""
    if not drug1 or not drug1.strip():
        raise ValueError("drug1 is required")
    if not drug2 or not drug2.strip():
        raise ValueError("drug2 is required")
    if not text or not text.strip():
        raise ValueError("text is required")

    logger.info("Stage 4 LLM prediction for %s vs %s", drug1, drug2)

    llm = get_structured_llm(model_name=model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(Stage4LLMPrediction)
    prompt = STAGE4_PREDICTION_PROMPT.format(drug1=drug1.strip(), drug2=drug2.strip(), text=text.strip())
    result = structured_llm.invoke(prompt)

    logger.info(
        "Stage 4 prediction complete: interaction=%s confidence=%.3f",
        result.interaction,
        result.confidence_score,
    )
    return result


def predict_llm_stage4_safe(
    drug1: str,
    drug2: str,
    text: str,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.0,
) -> Stage4LLMPrediction:
    """Safe Stage 4 wrapper that falls back to a conservative negative prediction."""
    try:
        return predict_llm_stage4(
            drug1=drug1,
            drug2=drug2,
            text=text,
            model_name=model_name,
            temperature=temperature,
        )
    except Exception as exc:
        logger.error("Stage 4 LLM prediction failed: %s", exc, exc_info=True)
        return Stage4LLMPrediction(
            interaction=False,
            confidence_score=0.0,
            evidence_snippet=str(exc)[:1000],
        )
