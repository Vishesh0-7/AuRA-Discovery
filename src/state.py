"""
Shared state definition for the Agentic Graph-RAG system.

This module defines the TypedDict that represents the shared state
passed between nodes in the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class AgentState(TypedDict, total=False):
    """
    Shared state for the agentic workflow.
    
    Attributes:
        query: User's initial research query
        papers: List of retrieved PubMed papers
        entities: Extracted biomedical entities (genes, proteins, compounds)
        relationships: Identified relationships between entities
        graph_data: Structured data ready for Neo4j insertion
        validation_results: Results from validation agents
        enrichment_data: Additional data from external APIs (ChEMBL, etc.)
        messages: Conversation/reasoning history
        current_step: Current workflow step identifier
        metadata: Additional contextual information
        error: Error message if any step fails
        timestamp: Timestamp of the current state update
    """
    
    # Core workflow data
    query: str
    papers: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_data: Dict[str, Any]
    
    # Validation and enrichment
    validation_results: Dict[str, Any]
    enrichment_data: Dict[str, Any]
    
    # Workflow tracking
    messages: List[Dict[str, str]]
    current_step: str
    metadata: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    timestamp: datetime


class EntityExtraction(TypedDict):
    """Schema for extracted entities."""
    entity_id: str
    entity_type: str  # gene, protein, compound, disease, etc.
    name: str
    aliases: List[str]
    confidence: float
    source_paper_id: str
    context: str


class Relationship(TypedDict):
    """Schema for entity relationships."""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    evidence: str
    source_paper_id: str


class PaperMetadata(TypedDict):
    """Schema for PubMed paper metadata."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    doi: str
    publication_date: str


class DiscoveryAgentState(TypedDict, total=False):
    """
    State for the Discovery Agent workflow with ChEMBL validation.
    
    This state tracks the complete validation pipeline from extraction
    to validation to Neo4j storage with conflict detection.
    
    **State Mutation Pattern:**
    This TypedDict is intentionally mutable to work with LangGraph's state
    management. Each node receives the state, modifies it in-place, and
    returns it. LangGraph handles state versioning and persistence.
    
    **Best Practices:**
    1. Always return the modified state from nodes (even though mutated in-place)
    2. Initialize all optional fields in process_paper() to avoid KeyErrors
    3. Use descriptive field names to track data flow through nodes
    4. Document which nodes modify which fields (see below)
    
    **State Flow by Node:**
    - node_extract: Sets extracted_ddis, next_step
    - node_validate: Sets validation_results, conflicts, discoveries, final_decision
    - node_flag_conflict: Modifies conflicts list with metadata
    - node_mark_discovery: Modifies discoveries list with metadata
    - node_update_graph: Sets graph_update_results
    
    Attributes:
        paper_id: DOI or identifier of the paper being processed
        paper_title: Title of the paper (stored but not currently used in extraction)
        raw_abstract: Original abstract text from the paper
        extracted_ddis: List of drug-drug interactions extracted by Gemini
        validation_results: ChEMBL validation results for each DDI
        conflicts: List of DDIs with contradictions between paper and ChEMBL
        discoveries: List of potential novel findings (unknown drugs/interactions)
        failed_validations: List of interactions that failed validation (error recovery)
        final_decision: Overall outcome ('validated', 'discoveries', 'conflicts', 'error')
        next_step: Next node to route to in the workflow
        graph_update_results: Results from Neo4j updates (includes failed_interactions)
        error: Error message if something fails
        metadata: Additional context (e.g., timestamps, processing info)
    """
    
    # Paper metadata
    paper_id: str
    paper_title: str  # Note: Currently stored but not used in extraction
    raw_abstract: str
    
    # Extraction results
    extracted_ddis: List[Dict[str, Any]]
    
    # Validation results
    validation_results: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    discoveries: List[Dict[str, Any]]
    failed_validations: List[Dict[str, Any]]  # Error recovery tracking
    
    # Workflow control
    next_step: str
    final_decision: str
    
    # Neo4j update results
    graph_update_results: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    
    # Additional context
    metadata: Dict[str, Any]
