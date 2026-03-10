"""
bioRxiv API Tool

Functions for searching and retrieving papers from bioRxiv.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class BioRxivClient:
    """
    Client for interacting with the bioRxiv API.
    
    Documentation: https://api.biorxiv.org/
    """
    
    BASE_URL = "https://api.biorxiv.org"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize bioRxiv client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    def search_papers(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search bioRxiv papers by query.
        
        Args:
            query: Search query string
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_results: Maximum number of results to return
        
        Returns:
            List of paper metadata dictionaries
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # bioRxiv API endpoint
        url = f"{self.BASE_URL}/details/biorxiv/{start_date}/{end_date}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Filter by query in title or abstract
            papers = data.get("collection", [])
            filtered_papers = [
                paper for paper in papers
                if query.lower() in paper.get("title", "").lower()
                or query.lower() in paper.get("abstract", "").lower()
            ]
            
            logger.info(f"Found {len(filtered_papers)} papers matching query: {query}")
            return filtered_papers[:max_results]
        
        except requests.RequestException as e:
            logger.error(f"Error fetching papers from bioRxiv: {e}")
            return []
    
    def get_paper_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific paper by DOI.
        
        Args:
            doi: Paper DOI
        
        Returns:
            Paper metadata dictionary or None if not found
        """
        url = f"{self.BASE_URL}/details/biorxiv/{doi}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            collection = data.get("collection", [])
            return collection[0] if collection else None
        
        except requests.RequestException as e:
            logger.error(f"Error fetching paper {doi}: {e}")
            return None
    
    def get_papers_by_category(
        self,
        category: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get recent papers from a specific category.
        
        Args:
            category: bioRxiv category (e.g., 'pharmacology-and-toxicology', 'bioinformatics')
            days_back: Number of days to look back
        
        Returns:
            List of paper metadata dictionaries
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Convert category name to URL slug format (spaces to hyphens, lowercase)
        category_slug = category.lower().replace(' ', '-').replace('_', '-')
        
        # Use category-specific endpoint for better filtering
        url = f"{self.BASE_URL}/details/biorxiv/{start_date}/{end_date}/0/{category_slug}"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            papers = data.get("collection", [])
            
            # Papers from category-specific endpoint are already filtered
            return papers
        
        except requests.RequestException as e:
            logger.error(f"Error fetching papers by category: {e}")
            return []


# Convenience functions for direct use

def search_biorxiv(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Quick search function for bioRxiv papers.
    
    Args:
        query: Search query
        max_results: Maximum results to return
    
    Returns:
        List of papers
    """
    client = BioRxivClient()
    return client.search_papers(query, max_results=max_results)


def get_latest_preprints(
    category: str = "pharmacology-and-toxicology",
    count: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch the latest preprints from bioRxiv for a specific category.
    
    This function retrieves recent papers from bioRxiv and formats them
    with standardized fields (doi, title, abstract, authors) suitable
    for ingestion into a knowledge graph.
    
    Args:
        category: bioRxiv category (default: 'pharmacology-and-toxicology')
                  Use hyphenated slug format for category names
        count: Maximum number of papers to return (default: 10)
    
    Returns:
        List of dictionaries, each containing:
            - doi (str): Paper DOI
            - title (str): Paper title
            - abstract (str): Paper abstract
            - authors (list): List of author names
            - category (str): bioRxiv category
            - date (str): Publication date
            - url (str): Full URL to the paper
    
    Example:
        >>> papers = get_latest_preprints(category='pharmacology-and-toxicology', count=50)
        >>> for paper in papers:
        ...     print(paper['title'])
    """
    logger.info(f"Fetching latest {count} preprints from category: {category}")
    
    client = BioRxivClient()
    
    # Fetch papers from the last 30 days
    papers = client.get_papers_by_category(category=category, days_back=30)
    
    if not papers:
        logger.warning(f"No papers found for category: {category}")
        return []
    
    # Sort by date (most recent first)
    sorted_papers = sorted(
        papers,
        key=lambda x: x.get("date", ""),
        reverse=True
    )
    
    # Format papers with standardized fields
    formatted_papers = []
    for paper in sorted_papers[:count]:
        try:
            # Parse authors string into list
            authors_str = paper.get("authors", "")
            authors_list = [
                author.strip()
                for author in authors_str.split(";")
                if author.strip()
            ]
            
            formatted_paper = {
                "doi": paper.get("doi", ""),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": authors_list,
                "category": paper.get("category", category),
                "date": paper.get("date", ""),
                "url": f"https://www.biorxiv.org/content/{paper.get('doi', '')}v1"
            }
            
            # Only add papers with required fields
            if formatted_paper["doi"] and formatted_paper["title"]:
                formatted_papers.append(formatted_paper)
                logger.debug(f"Formatted paper: {formatted_paper['doi']}")
            else:
                logger.warning(f"Skipping paper with missing required fields")
        
        except Exception as e:
            logger.error(f"Error formatting paper: {e}", exc_info=True)
            continue
    
    logger.info(f"Successfully retrieved and formatted {len(formatted_papers)} papers")
    return formatted_papers

