"""
PubMed API Tool

Functions for searching and retrieving papers from PubMed using NCBI E-utilities.
"""

import logging
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import time
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)


def retry_on_network_error(max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    Decorator to retry network requests on failure with exponential backoff.
    
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
                except (requests.RequestException, ET.ParseError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                        raise
            
            raise last_exception
        return wrapper
    return decorator


class PubMedClient:
    """
    Client for interacting with the PubMed API (NCBI E-utilities).
    
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, timeout: int = 30, email: Optional[str] = None):
        """
        Initialize PubMed client.
        
        Args:
            timeout: Request timeout in seconds
            email: Contact email (recommended by NCBI)
        """
        self.timeout = timeout
        self.email = email or "aura-discovery@example.com"
    
    @retry_on_network_error(max_retries=3, delay=2.0, backoff=2.0)
    def search_papers(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed papers and get PMIDs.
        
        Args:
            query: Search query string (PubMed query syntax)
            max_results: Maximum number of results to return
            start_date: Start date in YYYY/MM/DD format
            end_date: End date in YYYY/MM/DD format
        
        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "tool": "aura-discovery",
            "email": self.email
        }
        
        # Add date range if provided
        if start_date and end_date:
            params["datetype"] = "pdat"
            params["mindate"] = start_date
            params["maxdate"] = end_date
        
        try:
            response = requests.get(self.ESEARCH_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            id_list = root.find("IdList")
            
            if id_list is None:
                logger.warning("No results found")
                return []
            
            pmids = [id_elem.text for id_elem in id_list.findall("Id")]
            logger.info(f"Found {len(pmids)} papers matching query: {query}")
            return pmids
        
        except requests.RequestException as e:
            logger.error(f"Error fetching papers from PubMed: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed response: {e}")
            return []
    
    def fetch_papers(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch full paper details for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of paper metadata dictionaries
        """
        if not pmids:
            return []
        
        # NCBI recommends batches of 200 or fewer
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            papers = self._fetch_batch(batch)
            all_papers.extend(papers)
            
            # Be nice to NCBI servers - rate limit to 3 requests per second
            if i + batch_size < len(pmids):
                time.sleep(0.34)
        
        return all_papers
    
    @retry_on_network_error(max_retries=3, delay=2.0, backoff=2.0)
    def _fetch_batch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch a batch of papers.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of paper dictionaries
        """
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "aura-discovery",
            "email": self.email
        }
        
        try:
            response = requests.get(self.EFETCH_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            articles_found = root.findall(".//PubmedArticle")
            
            for article in articles_found:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
                else:
                    logger.warning("Failed to parse article, skipping")
            
            logger.info(f"Fetched {len(papers)} paper details from {len(articles_found)} articles")
            if len(papers) < len(articles_found):
                logger.warning(f"{len(articles_found) - len(papers)} articles failed to parse")
            return papers
        
        except requests.RequestException as e:
            logger.error(f"Error fetching paper details: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"Error parsing paper details: {e}")
            return []
    
    def _parse_article(self, article: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Parse a PubmedArticle XML element into a dictionary.
        
        Args:
            article: PubmedArticle XML element
        
        Returns:
            Paper dictionary or None if parsing fails
        """
        try:
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None
            
            # Get PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Get article metadata
            article_elem = medline.find(".//Article")
            if article_elem is None:
                return None
            
            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_parts = []
            abstract_elem = article_elem.find(".//Abstract")
            if abstract_elem is not None:
                for abstract_text in abstract_elem.findall(".//AbstractText"):
                    label = abstract_text.get("Label", "")
                    text = abstract_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            
            abstract = " ".join(abstract_parts)
            
            # Authors
            authors = []
            author_list = article_elem.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall(".//Author"):
                    last_name = author.find(".//LastName")
                    fore_name = author.find(".//ForeName")
                    if last_name is not None:
                        name = last_name.text
                        if fore_name is not None:
                            name = f"{fore_name.text} {name}"
                        authors.append(name)
            
            # Publication date with robust parsing and fallbacks
            pub_date = article_elem.find(".//Journal/JournalIssue/PubDate")
            date_str = ""
            if pub_date is not None:
                year = pub_date.find(".//Year")
                month = pub_date.find(".//Month")
                day = pub_date.find(".//Day")
                
                year_text = year.text.strip() if year is not None and year.text else ""
                month_text = month.text.strip() if month is not None and month.text else ""
                day_text = day.text.strip() if day is not None and day.text else ""
                
                if year_text:
                    try:
                        # Convert month name to number if needed
                        month_map = {
                            "Jan": "01", "January": "01",
                            "Feb": "02", "February": "02",
                            "Mar": "03", "March": "03",
                            "Apr": "04", "April": "04",
                            "May": "05",
                            "Jun": "06", "June": "06",
                            "Jul": "07", "July": "07",
                            "Aug": "08", "August": "08",
                            "Sep": "09", "September": "09",
                            "Oct": "10", "October": "10",
                            "Nov": "11", "November": "11",
                            "Dec": "12", "December": "12"
                        }
                        
                        # Get numeric month
                        if month_text:
                            month_num = month_map.get(month_text, month_text)
                            # Validate month is numeric and in range
                            try:
                                month_int = int(month_num)
                                if not (1 <= month_int <= 12):
                                    raise ValueError(f"Month out of range: {month_int}")
                            except ValueError as e:
                                logger.warning(f"Invalid month '{month_text}' for PMID {pmid}, using 01: {e}")
                                month_num = "01"
                        else:
                            month_num = "01"
                        
                        # Validate day is numeric and in reasonable range
                        if day_text:
                            try:
                                day_int = int(day_text)
                                if not (1 <= day_int <= 31):
                                    raise ValueError(f"Day out of range: {day_int}")
                            except ValueError as e:
                                logger.warning(f"Invalid day '{day_text}' for PMID {pmid}, using 01: {e}")
                                day_text = "01"
                        else:
                            day_text = "01"
                        
                        # Construct date string
                        date_str = f"{year_text}-{month_num.zfill(2)}-{day_text.zfill(2)}"
                        
                        # Validate final date format (basic check)
                        from datetime import datetime
                        try:
                            datetime.strptime(date_str, "%Y-%m-%d")
                        except ValueError:
                            # Fall back to year only
                            logger.warning(f"Invalid date '{date_str}' for PMID {pmid}, using year only")
                            date_str = year_text
                            
                    except Exception as e:
                        # Fall back to year only on any parsing error
                        logger.warning(f"Date parsing failed for PMID {pmid}, using year only: {e}")
                        date_str = year_text
            
            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # DOI
            doi = ""
            for article_id in medline.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # MeSH terms for categorization
            mesh_terms = []
            mesh_list = medline.find(".//MeshHeadingList")
            if mesh_list is not None:
                for mesh in mesh_list.findall(".//DescriptorName"):
                    mesh_terms.append(mesh.text)
            
            return {
                "pmid": pmid,
                "doi": doi or f"PMID:{pmid}",
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "date": date_str,
                "journal": journal,
                "mesh_terms": mesh_terms,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
        
        except Exception as e:
            logger.error(f"Error parsing article: {e}", exc_info=True)
            return None
    
    def search_and_fetch(
        self,
        query: str,
        max_results: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search and fetch papers in one call.
        
        Args:
            query: Search query
            max_results: Maximum results
            start_date: Start date in YYYY/MM/DD format
            end_date: End date in YYYY/MM/DD format
        
        Returns:
            List of paper dictionaries
        """
        pmids = self.search_papers(query, max_results, start_date, end_date)
        return self.fetch_papers(pmids)


# Convenience functions for direct use

def search_pubmed(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Quick search function for PubMed papers.
    
    Args:
        query: Search query
        max_results: Maximum results to return
    
    Returns:
        List of papers
    """
    client = PubMedClient()
    return client.search_and_fetch(query, max_results=max_results)


def get_latest_papers(
    query: str = "drug interaction[MeSH Terms] AND pharmacology[MeSH Terms]",
    count: int = 50,
    years_back: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Fetch the latest papers from PubMed for drug interaction research.
    
    This function retrieves recent papers from PubMed and formats them
    with standardized fields (doi, title, abstract, authors) suitable
    for ingestion into a knowledge graph.
    
    Args:
        query: PubMed search query (default focuses on drug interactions)
               Examples:
               - "drug interaction[MeSH Terms]"
               - "drug interaction AND pharmacology"
               - "cytochrome P450 AND drug metabolism"
        count: Maximum number of papers to return (default: 50)
         years_back: How many years of PubMed history to search.
                  If None, uses PUBMED_YEARS_BACK env var (default: 20).
                  Set to 0 or a negative value to disable date filtering.
    
    Returns:
        List of dictionaries, each containing:
            - pmid (str): PubMed ID
            - doi (str): Paper DOI or PMID reference
            - title (str): Paper title
            - abstract (str): Paper abstract
            - authors (list): List of author names
            - date (str): Publication date
            - journal (str): Journal name
            - mesh_terms (list): MeSH subject headings
            - url (str): Full URL to the paper
    
    Example:
        >>> papers = get_latest_papers(
        ...     query="drug interaction[MeSH Terms]",
        ...     count=50
        ... )
        >>> for paper in papers:
        ...     print(paper['title'])
    """
    logger.info(f"Fetching latest {count} papers from PubMed")
    logger.info(f"Query: {query}")
    
    client = PubMedClient()

    # Allow configurable historical backfill depth via env var.
    if years_back is None:
        raw_years_back = os.getenv("PUBMED_YEARS_BACK", "20")
        try:
            years_back = int(raw_years_back)
        except ValueError:
            logger.warning(
                "Invalid PUBMED_YEARS_BACK='%s'; falling back to 20",
                raw_years_back,
            )
            years_back = 20

    if years_back > 0:
        end_date = datetime.now().strftime("%Y/%m/%d")
        start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y/%m/%d")
        logger.info("Using PubMed date window: %s to %s (%d years)", start_date, end_date, years_back)
        papers = client.search_and_fetch(
            query=query,
            max_results=count,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        logger.info("Using PubMed search without date filter")
        papers = client.search_and_fetch(
            query=query,
            max_results=count,
        )
    
    if not papers:
        logger.warning(f"No papers found for query: {query}")
        return []
    
    # Sort by date (most recent first)
    sorted_papers = sorted(
        papers,
        key=lambda x: x.get("date", ""),
        reverse=True
    )
    
    # Filter papers with required fields
    valid_papers = []
    for paper in sorted_papers[:count]:
        if paper.get("doi") and paper.get("title") and paper.get("abstract"):
            valid_papers.append(paper)
            logger.debug(f"Formatted paper: {paper['pmid']}")
        else:
            logger.warning(f"Skipping paper {paper.get('pmid', 'unknown')} with missing required fields")
    
    logger.info(f"Successfully retrieved and formatted {len(valid_papers)} papers")
    return valid_papers
