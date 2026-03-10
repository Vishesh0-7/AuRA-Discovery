"""
Drug Interaction Extractor Agent

This module uses Google Gemini 1.5 Pro with structured output to extract
drug-drug interactions from biomedical literature abstracts.
"""

import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schema import ExtractionResult
from src.factory import get_structured_llm

# Configure logging
logger = logging.getLogger(__name__)


# Extraction prompt template
EXTRACTION_PROMPT = """You are a Ph.D. pharmacologist with expertise in drug interactions and pharmacokinetics.

Your task is to carefully analyze the following scientific abstract and extract ONLY clear, explicit drug-drug interactions.

EXTRACTION GUIDELINES:
1. Only extract interactions that are EXPLICITLY mentioned in the text
2. Focus on chemical compounds, drugs, and pharmaceutical agents
3. Identify the specific type of interaction (e.g., Inhibits, Potentiates, Toxic, Synergistic)
4. Include the evidence text that supports each interaction
5. Do NOT infer or speculate - only extract what is directly stated
6. If no clear interactions are found, return an empty list

INTERACTION TYPES TO LOOK FOR:
- Inhibits: One drug reduces the effect of another
- Potentiates: One drug enhances the effect of another
- Toxic: Combined use causes toxicity or adverse effects
- Synergistic: Drugs work together for enhanced therapeutic effect
- Antagonistic: Drugs have opposing effects
- Increases Risk: One drug increases risk when combined with another
- Contraindicated: Drugs should not be used together

PAPER ABSTRACT:
{abstract}

Extract all drug-drug interactions from this abstract. Be precise and conservative - only extract interactions that are clearly and explicitly stated."""


def extract_interactions(
    text: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.3,
    source_doi: Optional[str] = None
) -> ExtractionResult:
    """
    Extract drug interactions from text using Gemini with structured output.
    
    This function uses LangChain's ChatGoogleGenerativeAI with the 
    .with_structured_output() method to ensure the response matches
    the ExtractionResult Pydantic schema.
    
    Args:
        text: The text to analyze (typically a paper abstract)
        model_name: Gemini model to use (default: gemini-2.5-flash)
        temperature: Sampling temperature (lower = more focused, default: 0.3)
        source_doi: Optional DOI of the source paper for tracking
    
    Returns:
        ExtractionResult containing a list of Interaction objects
    
    Raises:
        ValueError: If text is empty or None
        Exception: If API call fails or structured output parsing fails
    
    Example:
        >>> abstract = "Study shows aspirin potentiates warfarin's anticoagulant effect..."
        >>> result = extract_interactions(abstract, source_doi="10.1101/2024.01.001")
        >>> print(f"Found {len(result)} interactions")
        >>> for interaction in result.interactions:
        ...     print(f"{interaction.drug_a} -> {interaction.drug_b}: {interaction.interaction_type}")
    """
    # Validate input
    if not text or not text.strip():
        logger.warning("Empty text provided for extraction")
        return ExtractionResult(
            interactions=[],
            source_doi=source_doi,
            extraction_notes="Empty or invalid text provided"
        )
    
    logger.info(f"Extracting interactions from text ({len(text)} chars)")
    if source_doi:
        logger.info(f"Source DOI: {source_doi}")
    
    try:
        # Get Gemini model configured for structured output
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=4096
        )
        
        # Configure for structured output using Pydantic schema
        structured_llm = llm.with_structured_output(ExtractionResult)
        
        # Format the prompt with the abstract
        prompt = EXTRACTION_PROMPT.format(abstract=text)
        
        # Invoke the model
        logger.debug("Calling Gemini API for structured extraction...")
        result = structured_llm.invoke(prompt)
        
        # Add source DOI if provided
        if source_doi and isinstance(result, ExtractionResult):
            result.source_doi = source_doi
        
        # Log results
        interaction_count = len(result.interactions) if hasattr(result, 'interactions') else 0
        logger.info(f"Extraction complete: Found {interaction_count} interactions")
        
        if interaction_count > 0:
            for idx, interaction in enumerate(result.interactions, 1):
                logger.debug(
                    f"  [{idx}] {interaction.drug_a} <-> {interaction.drug_b}: "
                    f"{interaction.interaction_type}"
                )
        
        return result
    
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        
        # Return empty result with error note
        return ExtractionResult(
            interactions=[],
            source_doi=source_doi,
            extraction_notes=f"Extraction failed: {str(e)}"
        )


def extract_interactions_batch(
    texts: list[str],
    source_dois: Optional[list[str]] = None,
    model_name: str = "gemini-2.5-flash"
) -> list[ExtractionResult]:
    """
    Extract interactions from multiple texts in batch.
    
    Note: This processes texts sequentially to respect API rate limits.
    For true parallel processing, consider using async version.
    
    Args:
        texts: List of texts to analyze
        source_dois: Optional list of DOIs corresponding to texts
        model_name: Gemini model to use
    
    Returns:
        List of ExtractionResult objects, one per text
    
    Example:
        >>> abstracts = ["text1...", "text2..."]
        >>> dois = ["10.1101/001", "10.1101/002"]
        >>> results = extract_interactions_batch(abstracts, dois)
        >>> total = sum(len(r.interactions) for r in results)
        >>> print(f"Total interactions: {total}")
    """
    logger.info(f"Batch extracting interactions from {len(texts)} texts")
    
    # Prepare DOI list if not provided
    if source_dois is None:
        source_dois = [None] * len(texts)
    elif len(source_dois) != len(texts):
        raise ValueError(
            f"Length mismatch: {len(texts)} texts but {len(source_dois)} DOIs provided"
        )
    
    results = []
    for idx, (text, doi) in enumerate(zip(texts, source_dois), 1):
        logger.info(f"Processing text {idx}/{len(texts)}")
        try:
            result = extract_interactions(
                text=text,
                model_name=model_name,
                source_doi=doi
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to extract from text {idx}: {e}")
            # Add empty result for this text to maintain index alignment
            results.append(
                ExtractionResult(
                    interactions=[],
                    source_doi=doi,
                    extraction_notes=f"Batch extraction failed: {str(e)}"
                )
            )
    
    total_interactions = sum(len(r.interactions) for r in results)
    logger.info(
        f"Batch extraction complete: {total_interactions} interactions "
        f"from {len(results)} texts"
    )
    
    return results


def validate_extraction(result: ExtractionResult) -> bool:
    """
    Validate that an extraction result is well-formed.
    
    Args:
        result: ExtractionResult to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(result, ExtractionResult):
        logger.error("Invalid result type")
        return False
    
    if not hasattr(result, 'interactions'):
        logger.error("Result missing interactions field")
        return False
    
    # Check each interaction
    for interaction in result.interactions:
        if not interaction.drug_a or not interaction.drug_b:
            logger.warning("Interaction missing drug names")
            return False
        
        if not interaction.interaction_type:
            logger.warning("Interaction missing type")
            return False
        
        if not interaction.evidence_text:
            logger.warning("Interaction missing evidence")
            return False
    
    return True
