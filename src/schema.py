"""
Pydantic Schema Definitions for Drug Interaction Extraction

This module defines the structured output schemas used by LLM extraction
for extracting drug interactions from biomedical literature.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Interaction(BaseModel):
    """
    Represents a single drug-drug interaction extracted from text.
    
    Attributes:
        drug_a: Name of the first drug/compound
        drug_b: Name of the second drug/compound
        interaction_type: Type of interaction (Inhibits, Potentiates, Toxic, etc.)
        evidence_text: The specific text from the paper that supports this interaction
        confidence: Optional confidence score (0.0 to 1.0)
    
    Example:
        >>> interaction = Interaction(
        ...     drug_a="Aspirin",
        ...     drug_b="Warfarin",
        ...     interaction_type="Potentiates",
        ...     evidence_text="Aspirin significantly enhances the anticoagulant effect of warfarin"
        ... )
    """
    
    drug_a: str = Field(
        ...,
        description="Name of the first drug, compound, or chemical entity",
        min_length=1,
        max_length=200
    )
    
    drug_b: str = Field(
        ...,
        description="Name of the second drug, compound, or chemical entity",
        min_length=1,
        max_length=200
    )
    
    interaction_type: str = Field(
        ...,
        description=(
            "Type of interaction between the drugs. Common types: "
            "'Inhibits', 'Potentiates', 'Toxic', 'Synergistic', 'Antagonistic', "
            "'Reduces Efficacy', 'Increases Risk', 'Contraindicated'"
        ),
        min_length=1,
        max_length=100
    )
    
    evidence_text: str = Field(
        ...,
        description="Direct quote or paraphrase from the source text supporting this interaction",
        min_length=10,
        max_length=1000
    )
    
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for this extraction (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "drug_a": "Metformin",
                "drug_b": "Contrast Dye",
                "interaction_type": "Increases Risk",
                "evidence_text": "Metformin use with iodinated contrast can lead to lactic acidosis",
                "confidence": 0.95
            }
        }


class ExtractionResult(BaseModel):
    """
    Container model for multiple drug interactions extracted from a single text.
    
    This is the top-level model returned by structured output,
    containing all interactions found in the analyzed text.
    
    Attributes:
        interactions: List of Interaction objects extracted from the text
        source_doi: Optional DOI of the source paper
        extraction_notes: Optional notes about the extraction process
    
    Example:
        >>> result = ExtractionResult(
        ...     interactions=[interaction1, interaction2],
        ...     source_doi="10.1101/2024.01.001",
        ...     extraction_notes="Found 2 clear drug interactions"
        ... )
    """
    
    interactions: List[Interaction] = Field(
        default_factory=list,
        description="List of drug interactions extracted from the text"
    )
    
    source_doi: Optional[str] = Field(
        default=None,
        description="DOI of the source paper (if available)"
    )
    
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Optional notes about the extraction quality or issues",
        max_length=500
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "interactions": [
                    {
                        "drug_a": "Warfarin",
                        "drug_b": "NSAIDs",
                        "interaction_type": "Increases Risk",
                        "evidence_text": "Concurrent use increases bleeding risk",
                        "confidence": 0.9
                    }
                ],
                "source_doi": "10.1101/2024.01.001",
                "extraction_notes": "High quality abstract with clear interactions"
            }
        }
    
    def __len__(self) -> int:
        """Return the number of interactions."""
        return len(self.interactions)
    
    def has_interactions(self) -> bool:
        """Check if any interactions were found."""
        return len(self.interactions) > 0


class DDIPrediction(BaseModel):
    """Binary DDI prediction returned by the LLM model A."""

    label: Literal["DDI", "No DDI"] = Field(
        ...,
        description="Binary prediction for whether the drug pair shows a drug-drug interaction",
    )

    confidence: float = Field(
        ...,
        description="Confidence score for the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

    rationale: Optional[str] = Field(
        default=None,
        description="Short explanation supporting the prediction",
        max_length=1000,
    )

    evidence_text: Optional[str] = Field(
        default=None,
        description="Relevant text span from the abstract used for the decision",
        max_length=1000,
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "label": "DDI",
                "confidence": 0.91,
                "rationale": "The abstract explicitly states the pair increases exposure and adverse events.",
                "evidence_text": "...coadministration significantly increased plasma concentration...",
            }
        }


class Stage4LLMPrediction(BaseModel):
    """Strict schema for Stage 4 LLM predictions."""

    interaction: bool = Field(
        ...,
        description="True when the abstract supports a DDI for the provided pair, otherwise False.",
    )

    confidence_score: float = Field(
        ...,
        description="Confidence score for interaction classification in the range [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )

    evidence_snippet: Optional[str] = Field(
        default=None,
        description="Short supporting snippet from the abstract, or null if no direct evidence exists.",
        max_length=1000,
    )

    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "interaction": True,
                "confidence_score": 0.87,
                "evidence_snippet": "Coadministration significantly increased plasma concentration and adverse events.",
            }
        }


# Alias for backward compatibility
ExtractionResults = ExtractionResult


# Predefined interaction types for validation (optional)
INTERACTION_TYPES = [
    "Inhibits",
    "Potentiates",
    "Toxic",
    "Synergistic",
    "Antagonistic",
    "Reduces Efficacy",
    "Increases Risk",
    "Contraindicated",
    "No Interaction",
    "Unknown"
]
