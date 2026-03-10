"""
ChEMBL Validator Tool

This module provides drug validation using the ChEMBL database.
Validates if drugs exist in ChEMBL and retrieves their mechanisms of action.
"""

import logging
from typing import Dict, Any, List, Optional
import time
import threading
from functools import wraps
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logging.warning("chembl_webresource_client not installed. Install with: pip install chembl_webresource_client")

from src.exceptions import ChEMBLAPIError, RateLimitError

# Configure logging
logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on exception with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} network error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                        raise ChEMBLAPIError(
                            f"API request failed after {max_retries + 1} attempts",
                            context={"function": func.__name__, "error": str(e)}
                        ) from e
                except Exception as e:
                    # Non-retryable errors, fail immediately
                    logger.error(f"{func.__name__} encountered non-retryable error: {e}")
                    raise
            
            # If all retries failed, raise the last exception
            if last_exception:
                raise ChEMBLAPIError(
                    f"API request failed after {max_retries + 1} attempts",
                    context={"function": func.__name__, "error": str(last_exception)}
                ) from last_exception
        return wrapper
    return decorator


class ChEMBLValidator:
    """
    Validator for checking drug existence and properties in ChEMBL database.
    
    This class provides methods to:
    - Search for drugs by name
    - Retrieve ChEMBL IDs
    - Get mechanisms of action
    - Handle rate limiting
    """
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize ChEMBL validator.
        
        Args:
            rate_limit_delay: Delay between API calls in seconds (default: 0.5s)
        """
        if not CHEMBL_AVAILABLE:
            raise ImportError(
                "chembl_webresource_client is required. "
                "Install with: pip install chembl_webresource_client"
            )
        
        self.molecule = new_client.molecule
        self.mechanism = new_client.mechanism
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0
        self._rate_limit_lock = threading.Lock()  # Thread-safe rate limiting
    
    def _rate_limit(self):
        """Apply rate limiting between API calls (thread-safe)."""
        with self._rate_limit_lock:
            current_time = time.time()
            elapsed = current_time - self._last_call_time
            
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
            
            self._last_call_time = time.time()
    
    @retry_on_error(max_retries=3, delay=1.0, backoff=2.0)
    def batch_search_drugs(self, drug_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Search for multiple drugs in ChEMBL in batch mode.
        
        This method efficiently searches for multiple drugs at once by:
        1. Deduplicating drug names (case-insensitive)
        2. Making one API call per unique drug
        3. Caching results for reuse
        
        Args:
            drug_names: List of drug names to search for
        
        Returns:
            Dictionary mapping drug name (lowercase) to validation results
        
        Example:
            >>> validator = ChEMBLValidator()
            >>> results = validator.batch_search_drugs(["Aspirin", "Warfarin", "Aspirin"])
            >>> print(results['aspirin']['chembl_id'])
            'CHEMBL25'
        """
        # Deduplicate drug names (case-insensitive)
        unique_drugs = {name.lower(): name for name in drug_names}
        results = {}
        
        logger.info(f"Batch searching {len(unique_drugs)} unique drugs from {len(drug_names)} total")
        
        for drug_lower, drug_original in unique_drugs.items():
            try:
                result = self.search_drug(drug_original)
                results[drug_lower] = result
            except Exception as e:
                logger.error(f"Failed to search drug '{drug_original}' in batch: {e}")
                results[drug_lower] = {
                    'status': 'error',
                    'query': drug_original,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'message': f"Batch search failed: {str(e)}"
                }
        
        return results
    
    @retry_on_error(max_retries=3, delay=1.0, backoff=2.0)
    def search_drug(self, drug_name: str) -> Dict[str, Any]:
        """
        Search for a drug in ChEMBL by name.
        
        Args:
            drug_name: Name of the drug to search for
        
        Returns:
            Dictionary with validation results:
                - status: 'found', 'unknown_molecule', or 'error'
                - chembl_id: ChEMBL identifier (if found)
                - preferred_name: Standard drug name (if found)
                - synonyms: List of alternative names (if found)
                - mechanisms: List of mechanisms of action (if found)
                - max_phase: Maximum clinical trial phase (if available)
                - molecule_type: Type of molecule (e.g., 'Small molecule')
                - error: Error message (if error occurred)
        
        Example:
            >>> validator = ChEMBLValidator()
            >>> result = validator.search_drug("Aspirin")
            >>> print(result['chembl_id'])
            'CHEMBL25'
        """
        logger.info(f"Searching ChEMBL for drug: {drug_name}")
        
        try:
            self._rate_limit()
            
            # Search for the molecule
            results = self.molecule.search(drug_name)
            
            if not results:
                logger.warning(f"Drug not found in ChEMBL: {drug_name}")
                return {
                    'status': 'unknown_molecule',
                    'query': drug_name,
                    'message': f"No results found for '{drug_name}' in ChEMBL"
                }
            
            # Get the best match (first result)
            molecule = results[0]
            chembl_id = molecule.get('molecule_chembl_id')
            
            # Get mechanisms of action
            mechanisms = self._get_mechanisms(chembl_id)
            
            # Extract synonyms
            synonyms = molecule.get('molecule_synonyms', [])
            if isinstance(synonyms, list):
                synonym_names = [syn.get('molecule_synonym') for syn in synonyms if isinstance(syn, dict)]
            else:
                synonym_names = []
            
            result = {
                'status': 'found',
                'query': drug_name,
                'chembl_id': chembl_id,
                'preferred_name': molecule.get('pref_name', drug_name),
                'synonyms': synonym_names[:5],  # Limit to top 5 synonyms
                'mechanisms': mechanisms,
                'max_phase': molecule.get('max_phase'),
                'molecule_type': molecule.get('molecule_type'),
                'first_approval': molecule.get('first_approval'),
                'therapeutic_flag': molecule.get('therapeutic_flag', False)
            }
            
            logger.info(f"Found drug in ChEMBL: {chembl_id} ({result['preferred_name']})") 
            return result
        
        except ChEMBLAPIError:
            # Re-raise our custom API errors (from retry decorator)
            raise
        except Exception as e:
            error_msg = f"Unexpected error searching ChEMBL for '{drug_name}'"
            logger.error(f"{error_msg}: {type(e).__name__} - {e}", exc_info=True)
            return {
                'status': 'error',
                'query': drug_name,
                'error': str(e),
                'error_type': type(e).__name__,
                'message': f"{error_msg}: {str(e)}"
            }
    
    def _get_mechanisms(self, chembl_id: str) -> List[Dict[str, Any]]:
        """
        Get mechanisms of action for a ChEMBL molecule.
        
        Args:
            chembl_id: ChEMBL molecule identifier
        
        Returns:
            List of mechanism dictionaries with action_type and target information
        """
        try:
            self._rate_limit()
            
            mechanisms = self.mechanism.filter(molecule_chembl_id=chembl_id)
            
            mechanism_list = []
            for mech in mechanisms[:10]:  # Limit to top 10 mechanisms
                mechanism_list.append({
                    'action_type': mech.get('action_type', 'Unknown'),
                    'mechanism_of_action': mech.get('mechanism_of_action', ''),
                    'target_name': mech.get('target_name', ''),
                    'direct_interaction': mech.get('direct_interaction', False)
                })
            
            return mechanism_list
        
        except Exception as e:
            logger.warning(f"Could not retrieve mechanisms for {chembl_id}: {e}")
            return []
    
    @retry_on_error(max_retries=2, delay=0.5, backoff=2.0)
    def validate_interaction(
        self,
        drug_a: str,
        drug_b: str,
        interaction_type: str,
        evidence_text: str
    ) -> Dict[str, Any]:
        """
        Validate a drug-drug interaction by checking both drugs in ChEMBL.
        
        Args:
            drug_a: Name of first drug
            drug_b: Name of second drug
            interaction_type: Type of interaction (e.g., 'Inhibits', 'Potentiates')
            evidence_text: Evidence text from paper
        
        Returns:
            Dictionary with validation results for both drugs and conflict detection:
                - drug_a_validation: ChEMBL results for drug A
                - drug_b_validation: ChEMBL results for drug B
                - both_known: Boolean indicating if both drugs are in ChEMBL
                - potential_discovery: Boolean indicating if this might be a new finding
                - conflict_detected: Boolean indicating contradictions with known data
                - validation_status: 'validated', 'partial', 'unknown', or 'conflict'
        """
        logger.info(f"Validating interaction: {drug_a} <-> {drug_b} ({interaction_type})")
        
        # Validate both drugs
        drug_a_result = self.search_drug(drug_a)
        drug_b_result = self.search_drug(drug_b)
        
        # Determine validation status
        both_known = (
            drug_a_result['status'] == 'found' and 
            drug_b_result['status'] == 'found'
        )
        
        one_unknown = (
            drug_a_result['status'] == 'unknown_molecule' or 
            drug_b_result['status'] == 'unknown_molecule'
        )
        
        # Check for contradictions between paper and ChEMBL data
        conflict_detected = False
        conflict_reason = None
        
        if both_known:
            # Basic conflict detection based on mechanisms of action
            drug_a_mechanisms = drug_a_result.get('mechanisms', [])
            drug_b_mechanisms = drug_b_result.get('mechanisms', [])
            
            # Extract action types
            drug_a_actions = set(m.get('action_type', '') for m in drug_a_mechanisms)
            drug_b_actions = set(m.get('action_type', '') for m in drug_b_mechanisms)
            
            # Check for contradictory claims
            harmful_types = ['Toxic', 'Contraindicated', 'Antagonistic', 'Increases Risk']
            beneficial_types = ['Synergistic', 'Potentiates']
            
            # If paper claims harmful interaction but both drugs are therapeutic
            if interaction_type in harmful_types:
                # Check if both drugs have therapeutic approval
                drug_a_therapeutic = drug_a_result.get('therapeutic_flag', False)
                drug_b_therapeutic = drug_b_result.get('therapeutic_flag', False)
                
                # Note: This is a simplified heuristic. In a real system, we'd check
                # actual drug-drug interaction databases or known contraindications.
                # For now, we flag low confidence but don't mark as conflict
                # since absence of evidence is not evidence of absence.
                if drug_a_therapeutic and drug_b_therapeutic:
                    logger.info(
                        f"Paper reports {interaction_type} between two therapeutic drugs. "
                        "Consider for manual review."
                    )
                    # Don't set conflict_detected=True as this requires domain expert review
        
        # Determine overall validation status
        if both_known and not conflict_detected:
            validation_status = 'validated'
            potential_discovery = False
        elif both_known and conflict_detected:
            validation_status = 'conflict'
            potential_discovery = False
        elif one_unknown:
            validation_status = 'partial'
            potential_discovery = True  # Unknown drug might be novel
        else:
            validation_status = 'unknown'
            potential_discovery = True
        
        result = {
            'drug_a': drug_a,
            'drug_b': drug_b,
            'interaction_type': interaction_type,
            'drug_a_validation': drug_a_result,
            'drug_b_validation': drug_b_result,
            'both_known': both_known,
            'potential_discovery': potential_discovery,
            'conflict_detected': conflict_detected,
            'conflict_reason': conflict_reason,
            'validation_status': validation_status,
            'evidence_text': evidence_text
        }
        
        logger.info(
            f"Validation complete: {validation_status} "
            f"(both_known={both_known}, potential_discovery={potential_discovery})"
        )
        
        return result


# Convenience functions

def validate_drug(drug_name: str) -> Dict[str, Any]:
    """
    Quick function to validate a single drug.
    
    Args:
        drug_name: Name of drug to validate
    
    Returns:
        Validation result dictionary
    """
    validator = ChEMBLValidator()
    return validator.search_drug(drug_name)


def validate_ddi(
    drug_a: str,
    drug_b: str,
    interaction_type: str,
    evidence_text: str = ""
) -> Dict[str, Any]:
    """
    Quick function to validate a drug-drug interaction.
    
    Args:
        drug_a: First drug name
        drug_b: Second drug name
        interaction_type: Type of interaction
        evidence_text: Supporting evidence
    
    Returns:
        Validation result dictionary
    """
    validator = ChEMBLValidator()
    return validator.validate_interaction(drug_a, drug_b, interaction_type, evidence_text)
