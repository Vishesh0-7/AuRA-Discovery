"""
ChEMBL API Tool

Functions for querying chemical and drug information from ChEMBL.
"""

import requests
from typing import List, Dict, Any, Optional


class ChEMBLClient:
    """
    Client for interacting with the ChEMBL REST API.
    
    Documentation: https://chembl.gitbook.io/chembl-interface-documentation/web-services
    """
    
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize ChEMBL client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.headers = {"Accept": "application/json"}
    
    def search_compound(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for compounds by name or identifier.
        
        Args:
            query: Compound name or identifier
        
        Returns:
            List of compound data
        """
        url = f"{self.BASE_URL}/molecule/search"
        params = {"q": query, "format": "json"}
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("molecules", [])
        
        except requests.RequestException as e:
            print(f"Error searching ChEMBL: {e}")
            return []
    
    def get_compound_by_chembl_id(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed compound information by ChEMBL ID.
        
        Args:
            chembl_id: ChEMBL compound identifier (e.g., 'CHEMBL25')
        
        Returns:
            Compound data dictionary or None
        """
        url = f"{self.BASE_URL}/molecule/{chembl_id}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            print(f"Error fetching compound {chembl_id}: {e}")
            return None
    
    def get_compound_targets(self, chembl_id: str) -> List[Dict[str, Any]]:
        """
        Get biological targets for a compound.
        
        Args:
            chembl_id: ChEMBL compound identifier
        
        Returns:
            List of target data
        """
        url = f"{self.BASE_URL}/activity"
        params = {
            "molecule_chembl_id": chembl_id,
            "format": "json",
            "limit": 100
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("activities", [])
        
        except requests.RequestException as e:
            print(f"Error fetching targets for {chembl_id}: {e}")
            return []
    
    def search_target(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for biological targets (proteins, genes).
        
        Args:
            query: Target name or identifier
        
        Returns:
            List of target data
        """
        url = f"{self.BASE_URL}/target/search"
        params = {"q": query, "format": "json"}
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("targets", [])
        
        except requests.RequestException as e:
            print(f"Error searching targets: {e}")
            return []
    
    def get_compound_mechanisms(self, chembl_id: str) -> List[Dict[str, Any]]:
        """
        Get mechanism of action data for a compound.
        
        Args:
            chembl_id: ChEMBL compound identifier
        
        Returns:
            List of mechanism data
        """
        url = f"{self.BASE_URL}/mechanism"
        params = {
            "molecule_chembl_id": chembl_id,
            "format": "json"
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("mechanisms", [])
        
        except requests.RequestException as e:
            print(f"Error fetching mechanisms for {chembl_id}: {e}")
            return []


# Convenience functions

def search_chembl_compound(query: str) -> List[Dict[str, Any]]:
    """
    Quick search for compounds in ChEMBL.
    
    Args:
        query: Compound name or identifier
    
    Returns:
        List of compounds
    """
    client = ChEMBLClient()
    return client.search_compound(query)


def get_compound_info(chembl_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed compound information.
    
    Args:
        chembl_id: ChEMBL identifier
    
    Returns:
        Compound data
    """
    client = ChEMBLClient()
    return client.get_compound_by_chembl_id(chembl_id)
