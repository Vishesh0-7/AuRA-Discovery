"""
Discovery Agent - Agentic Workflow for DDI Validation

This module implements a LangGraph workflow that:
1. Extracts drug-drug interactions from paper abstracts using Gemini
2. Validates each DDI against ChEMBL database
3. Detects conflicts and potential discoveries
4. Updates Neo4j with validated, flagged, or novel interactions

The agent uses conditional routing to handle different validation outcomes.
"""

import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from src.state import DiscoveryAgentState
from src.agents.extractor import extract_interactions
from src.tools.chembl_validator import ChEMBLValidator
from src.database.graph_connector import ResearchGraph
from src.exceptions import ExtractionError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)


class DiscoveryAgent:
    """
    Agentic workflow for discovering and validating drug-drug interactions.
    
    This agent orchestrates a multi-step process:
    - Extract DDIs from scientific abstracts
    - Validate against ChEMBL database
    - Detect conflicts with known pharmacology
    - Identify potential novel discoveries
    - Update Neo4j with appropriate labels
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Discovery Agent.
        
        Args:
            model_name: Gemini model to use for extraction
        """
        self.model_name = model_name
        self.validator = ChEMBLValidator()
        self.graph = None  # Will be initialized per workflow run
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        logger.info("Discovery Agent initialized")
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with all nodes and edges.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(DiscoveryAgentState)
        
        # Add nodes
        workflow.add_node("extract", self.node_extract)
        workflow.add_node("validate", self.node_validate)
        workflow.add_node("update_graph", self.node_update_graph)
        workflow.add_node("flag_conflict", self.node_flag_conflict)
        workflow.add_node("mark_discovery", self.node_mark_discovery)
        
        # Set entry point
        workflow.set_entry_point("extract")
        
        # Add edges with conditional routing
        workflow.add_edge("extract", "validate")
        
        # Conditional routing after validation
        workflow.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "conflicts": "flag_conflict",
                "discoveries": "mark_discovery",
                "validated": "update_graph",
                "end": END
            }
        )
        
        # All paths lead to graph update
        workflow.add_edge("flag_conflict", "update_graph")
        workflow.add_edge("mark_discovery", "update_graph")
        workflow.add_edge("update_graph", END)
        
        return workflow.compile()
    
    def node_extract(self, state: DiscoveryAgentState) -> DiscoveryAgentState:
        """
        Node 1: Extract drug-drug interactions from abstract using Gemini.
        
        Args:
            state: Current workflow state with raw_abstract
        
        Returns:
            Updated state with extracted_ddis populated
        """
        logger.info(f"[NODE: Extract] Processing paper: {state['paper_id']}")
        
        try:
            # Validate input
            if not state.get('raw_abstract'):
                raise ValidationError(
                    "Abstract text is empty or missing",
                    context={"paper_id": state['paper_id']}
                )
            
            # Extract interactions using Gemini
            result = extract_interactions(
                text=state['raw_abstract'],
                model_name=self.model_name,
                source_doi=state['paper_id']
            )
            
            # Convert to list of dictionaries
            extracted_ddis = [
                {
                    'drug_a': interaction.drug_a,
                    'drug_b': interaction.drug_b,
                    'interaction_type': interaction.interaction_type,
                    'evidence_text': interaction.evidence_text,
                    'confidence': interaction.confidence
                }
                for interaction in result.interactions
            ]
            
            state['extracted_ddis'] = extracted_ddis
            state['next_step'] = 'validate'
            
            logger.info(f"[NODE: Extract] Found {len(extracted_ddis)} interactions")
        
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            error_msg = f"Extraction failed for paper {state['paper_id']}: {type(e).__name__}"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            state['error'] = f"{error_msg}: {str(e)}"
            state['next_step'] = 'end'
            state['extracted_ddis'] = []  # Set empty list for downstream nodes
        
        return state
    
    def node_validate(self, state: DiscoveryAgentState) -> DiscoveryAgentState:
        """
        Node 2: Validate each extracted DDI against ChEMBL database with batch processing.
        
        This method uses batch processing to minimize API calls:
        1. Collects all unique drug names from interactions
        2. Performs batch ChEMBL lookup once
        3. Uses cached results to validate each interaction
        
        Args:
            state: State with extracted_ddis
        
        Returns:
            State with validation_results, conflicts, and discoveries populated
        """
        logger.info(f"[NODE: Validate] Validating {len(state['extracted_ddis'])} interactions")
        
        validation_results = []
        conflicts = []
        discoveries = []
        failed_validations = []  # Track failures for error recovery
        
        # PERFORMANCE OPTIMIZATION: Batch process drug lookups
        # Collect all unique drug names from interactions
        all_drugs = []
        for ddi in state['extracted_ddis']:
            all_drugs.extend([ddi['drug_a'], ddi['drug_b']])
        
        # Batch search all drugs (automatically deduplicated)
        try:
            logger.info(f"[NODE: Validate] Batch searching {len(set(all_drugs))} unique drugs")
            drug_cache = self.validator.batch_search_drugs(all_drugs)
            logger.info(f"[NODE: Validate] Drug cache populated with {len(drug_cache)} entries")
        except Exception as e:
            logger.error(f"[NODE: Validate] Batch drug search failed: {e}", exc_info=True)
            drug_cache = {}  # Fall back to empty cache, will search individually
        
        # Process each interaction with cached drug data
        for idx, ddi in enumerate(state['extracted_ddis']):
            try:
                drug_a_lower = ddi['drug_a'].lower()
                drug_b_lower = ddi['drug_b'].lower()
                
                # Get drug info from cache or search individually
                drug_a_result = drug_cache.get(drug_a_lower)
                drug_b_result = drug_cache.get(drug_b_lower)
                
                if drug_a_result is None:
                    logger.warning(f"Drug A '{ddi['drug_a']}' not in cache, searching individually")
                    drug_a_result = self.validator.search_drug(ddi['drug_a'])
                
                if drug_b_result is None:
                    logger.warning(f"Drug B '{ddi['drug_b']}' not in cache, searching individually")
                    drug_b_result = self.validator.search_drug(ddi['drug_b'])
                
                # Construct validation result
                both_known = (
                    drug_a_result['status'] == 'found' and 
                    drug_b_result['status'] == 'found'
                )
                
                one_unknown = (
                    drug_a_result['status'] == 'unknown_molecule' or 
                    drug_b_result['status'] == 'unknown_molecule'
                )
                
                # Basic conflict detection based on mechanisms
                conflict_detected = False
                conflict_reason = None
                
                if both_known:
                    # Check for mechanism contradictions
                    drug_a_mechanisms = drug_a_result.get('mechanisms', [])
                    drug_b_mechanisms = drug_b_result.get('mechanisms', [])
                    # Add conflict detection logic here if needed
                
                # Determine status
                if conflict_detected:
                    validation_status = 'conflict'
                elif both_known:
                    validation_status = 'validated'
                elif one_unknown:
                    validation_status = 'partial'
                else:
                    validation_status = 'unknown'
                
                potential_discovery = one_unknown and not conflict_detected
                
                result = {
                    'drug_a': ddi['drug_a'],
                    'drug_b': ddi['drug_b'],
                    'interaction_type': ddi['interaction_type'],
                    'evidence_text': ddi['evidence_text'],
                    'drug_a_validation': drug_a_result,
                    'drug_b_validation': drug_b_result,
                    'both_known': both_known,
                    'potential_discovery': potential_discovery,
                    'conflict_detected': conflict_detected,
                    'conflict_reason': conflict_reason,
                    'validation_status': validation_status
                }
                
                validation_results.append(result)
                
                # Categorize based on validation outcome
                if conflict_detected:
                    conflicts.append(result)
                    logger.warning(
                        f"[NODE: Validate] Conflict detected: "
                        f"{ddi['drug_a']} <-> {ddi['drug_b']}"
                    )
                
                if potential_discovery:
                    discoveries.append(result)
                    logger.info(
                        f"[NODE: Validate] Potential discovery: "
                        f"{ddi['drug_a']} <-> {ddi['drug_b']}"
                    )
            
            except Exception as e:
                # Error recovery: log failure but continue with remaining interactions
                error_context = {
                    'interaction_index': idx,
                    'drug_a': ddi.get('drug_a', 'unknown'),
                    'drug_b': ddi.get('drug_b', 'unknown'),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                failed_validations.append(error_context)
                logger.error(
                    f"[NODE: Validate] Failed to validate interaction {idx + 1}: "
                    f"{ddi.get('drug_a', '?')} <-> {ddi.get('drug_b', '?')}: {e}",
                    exc_info=True
                )
        
        # Update state after processing all interactions
        state['validation_results'] = validation_results
        state['conflicts'] = conflicts
        state['discoveries'] = discoveries
        state['failed_validations'] = failed_validations  # Track partial failures
        
        # Determine next step - prioritize conflicts, but also process discoveries
        # Note: Both flags could be set on different interactions
        if conflicts and discoveries:
            state['next_step'] = 'conflicts'  # Process conflicts first
            state['final_decision'] = 'conflicts_and_discoveries'
        elif conflicts:
            state['next_step'] = 'conflicts'
            state['final_decision'] = 'conflicts_detected'
        elif discoveries:
            state['next_step'] = 'discoveries'
            state['final_decision'] = 'potential_discoveries'
        elif validation_results:
            state['next_step'] = 'validated'
            state['final_decision'] = 'all_validated'
        else:
            state['next_step'] = 'end'
            state['final_decision'] = 'no_interactions'
        
        success_count = len(validation_results)
        failure_count = len(failed_validations)
        total_count = success_count + failure_count
        
        logger.info(
            f"[NODE: Validate] Complete: {success_count}/{total_count} validated "
            f"({len(conflicts)} conflicts, {len(discoveries)} discoveries, "
            f"{failure_count} failures)"
        )
        
        # If all validations failed, set error state
        if success_count == 0 and failure_count > 0:
            state['error'] = f"All {failure_count} validations failed"
            state['next_step'] = 'end'
        
        return state
    
    def node_flag_conflict(self, state: DiscoveryAgentState) -> DiscoveryAgentState:
        """
        Node 3a: Flag conflicts with special Neo4j properties.
        
        Adds conflict metadata to interactions that contradict ChEMBL data.
        
        Args:
            state: State with conflicts list
        
        Returns:
            State with conflict flags added to validation results
        """
        logger.info(f"[NODE: Flag Conflict] Processing {len(state['conflicts'])} conflicts")
        
        try:
            for conflict in state['conflicts']:
                # Add conflict metadata
                conflict['neo4j_labels'] = ['CONFLICT']
                conflict['conflict_flag'] = True
                conflict['conflict_color'] = 'red'
                conflict['needs_review'] = True
                
                logger.warning(
                    f"[NODE: Flag Conflict] Flagged: "
                    f"{conflict['drug_a']} <-> {conflict['drug_b']}"
                )
            
            state['next_step'] = 'update_graph'
        
        except Exception as e:
            logger.error(f"[NODE: Flag Conflict] Error: {e}", exc_info=True)
            state['error'] = str(e)
        
        return state
    
    def node_mark_discovery(self, state: DiscoveryAgentState) -> DiscoveryAgentState:
        """
        Node 3b: Mark potential discoveries with special labels.
        
        Identifies interactions involving unknown drugs or novel combinations.
        
        Args:
            state: State with discoveries list
        
        Returns:
            State with discovery flags added to validation results
        """
        logger.info(f"[NODE: Mark Discovery] Processing {len(state['discoveries'])} discoveries")
        
        try:
            for discovery in state['discoveries']:
                # Add discovery metadata
                discovery['neo4j_labels'] = ['POTENTIAL_DISCOVERY']
                discovery['discovery_flag'] = True
                discovery['discovery_color'] = 'gold'
                discovery['requires_validation'] = True
                
                logger.info(
                    f"[NODE: Mark Discovery] Marked: "
                    f"{discovery['drug_a']} <-> {discovery['drug_b']}"
                )
            
            state['next_step'] = 'update_graph'
        
        except Exception as e:
            logger.error(f"[NODE: Mark Discovery] Error: {e}", exc_info=True)
            state['error'] = str(e)
        
        return state
    
    def node_update_graph(self, state: DiscoveryAgentState) -> DiscoveryAgentState:
        """
        Node 4: Update Neo4j with all validated, flagged, and novel interactions.
        
        Args:
            state: State with validation_results
        
        Returns:
            State with graph_update_results populated
        """
        logger.info("[NODE: Update Graph] Writing results to Neo4j")
        
        update_results = {
            'validated': 0,
            'conflicts': 0,
            'discoveries': 0,
            'errors': 0,
            'failed_interactions': []  # Track which interactions failed
        }
        
        # Use context manager to ensure connection is closed
        try:
            with ResearchGraph() as graph:
                
                for idx, result in enumerate(state['validation_results']):
                    try:
                        # Determine the relationship type and properties
                        interaction_data = {
                            'drug_a': result['drug_a'],
                            'drug_b': result['drug_b'],
                            'interaction_type': result['interaction_type'],
                            'evidence_text': result['evidence_text'],
                            'source_doi': state['paper_id'],
                            'validation_status': result['validation_status'],
                            'chembl_validated': result['both_known'],
                            'drug_a_validated': result['drug_a_validation']['status'] == 'found',
                            'drug_b_validated': result['drug_b_validation']['status'] == 'found'
                        }
                        
                        # Add conflict or discovery flags
                        if result.get('conflict_flag'):
                            interaction_data['conflict_detected'] = True
                            interaction_data['needs_review'] = True
                            update_results['conflicts'] += 1
                        elif result.get('discovery_flag'):
                            interaction_data['potential_discovery'] = True
                            interaction_data['requires_validation'] = True
                            update_results['discoveries'] += 1
                        else:
                            update_results['validated'] += 1
                        
                        # Store in Neo4j with validation metadata
                        graph.upsert_drug_interaction(
                            drug_a=interaction_data['drug_a'],
                            drug_b=interaction_data['drug_b'],
                            interaction_type=interaction_data['interaction_type'],
                            evidence_text=interaction_data['evidence_text'],
                            source_doi=interaction_data['source_doi'],
                            validation_status=interaction_data.get('validation_status'),
                            chembl_validated=interaction_data.get('chembl_validated', False),
                            drug_a_validated=interaction_data.get('drug_a_validated', False),
                            drug_b_validated=interaction_data.get('drug_b_validated', False),
                            conflict_detected=interaction_data.get('conflict_detected', False),
                            potential_discovery=interaction_data.get('potential_discovery', False),
                            needs_review=interaction_data.get('needs_review', False),
                            requires_validation=interaction_data.get('requires_validation', False)
                        )
                        
                    except Exception as e:
                        # Error recovery: log failure but continue with remaining interactions
                        error_context = {
                            'interaction_index': idx,
                            'drug_a': result.get('drug_a', 'unknown'),
                            'drug_b': result.get('drug_b', 'unknown'),
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                        update_results['failed_interactions'].append(error_context)
                        update_results['errors'] += 1
                        logger.error(
                            f"[NODE: Update Graph] Failed to store interaction {idx + 1}: "
                            f"{result.get('drug_a', '?')} <-> {result.get('drug_b', '?')}: "
                            f"{type(e).__name__}: {e}"
                        )
            
            state['graph_update_results'] = update_results
            state['next_step'] = 'end'
            
            success_count = update_results['validated'] + update_results['conflicts'] + update_results['discoveries']
            total_attempted = success_count + update_results['errors']
            
            logger.info(
                f"[NODE: Update Graph] Complete: {success_count}/{total_attempted} stored "
                f"({update_results['validated']} validated, {update_results['conflicts']} conflicts, "
                f"{update_results['discoveries']} discoveries, {update_results['errors']} errors)"
            )
        
        except Exception as e:
            error_msg = f"Critical error in node_update_graph: {type(e).__name__}"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            state['error'] = f"{error_msg}: {str(e)}"
            state['graph_update_results'] = update_results  # Save partial results
        
        return state
    
    def _route_after_validation(self, state: DiscoveryAgentState) -> str:
        """
        Conditional routing logic after validation.
        
        Args:
            state: Current state with next_step determined
        
        Returns:
            Next node to route to: 'conflicts', 'discoveries', 'validated', or 'end'
        """
        next_step = state.get('next_step', 'end')
        logger.info(f"[ROUTER] Routing to: {next_step}")
        return next_step
    
    def process_paper(
        self,
        paper_id: str,
        abstract: str,
        title: str = ""
    ) -> Dict[str, Any]:
        """
        Process a single paper through the complete validation workflow.
        
        Args:
            paper_id: DOI or identifier of the paper
            abstract: Abstract text to analyze
            title: Optional paper title (stored in state for future use;
                   currently not used in extraction but available for
                   enhanced prompting or filtering)
        
        Returns:
            Final state after workflow completion with all results
        
        Note:
            The title parameter is stored in state['paper_title'] but not
            currently used in the Gemini extraction prompt. It's preserved
            for potential future enhancements like:
            - Title-based relevance filtering
            - Enhanced context for extraction
            - Metadata enrichment
        """
        logger.info(f"Processing paper: {paper_id}")
        
        # Initialize state
        initial_state: DiscoveryAgentState = {
            'paper_id': paper_id,
            'raw_abstract': abstract,
            'paper_title': title,
            'extracted_ddis': [],
            'validation_results': [],
            'conflicts': [],
            'discoveries': [],
            'next_step': 'extract',
            'final_decision': '',
            'graph_update_results': {},
            'error': None,
            'metadata': {}
        }
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            logger.info(
                f"Workflow complete for {paper_id}: {final_state.get('final_decision')}"
            )
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow failed for {paper_id}: {e}", exc_info=True)
            return {
                **initial_state,
                'error': str(e),
                'final_decision': 'error'
            }


# Convenience function for direct use
def discover_and_validate(
    paper_id: str,
    abstract: str,
    title: str = "",
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Run the complete discovery and validation workflow on a paper.
    
    Args:
        paper_id: Paper DOI or identifier
        abstract: Paper abstract text
        title: Paper title (optional, currently stored but not used in extraction)
        model_name: Gemini model to use
    
    Returns:
        Final workflow state with all results
    
    Note:
        The title is passed through to process_paper() where it's stored
        in the state but not currently used in the extraction logic.
        See process_paper() documentation for details.
    """
    agent = DiscoveryAgent(model_name=model_name)
    return agent.process_paper(paper_id, abstract, title)
