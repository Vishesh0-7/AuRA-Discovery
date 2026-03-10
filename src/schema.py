"""
Pydantic Schema Definitions for Drug Interaction Extraction

This module defines the structured output schemas used by Gemini 1.5 Pro
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
    
    This is the top-level model returned by Gemini's structured output,
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
