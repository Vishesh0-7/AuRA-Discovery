"""
Graph Connector for Research Paper Ingestion

This module provides the ResearchGraph class for managing the ingestion
of research papers and their relationships into a Neo4j knowledge graph.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, SessionExpired
from dotenv import load_dotenv
from src.exceptions import (
    DatabaseError,
    ConnectionError,
    QueryError,
    MissingCredentialsError,
    ValidationError
)

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ResearchGraph:
    """
    Neo4j graph database connector for research paper management.
    
    This class handles the ingestion of research papers, authors, and their
    relationships into a Neo4j knowledge graph. Papers are tracked with a
    processing status to support multi-phase workflows.
    
    Architectural Decisions:
    ----------------------
    1. **Context Manager Pattern**: This class provides __enter__/__exit__ 
       for automatic resource cleanup. Use with `with ResearchGraph() as graph:` 
       to ensure connections are properly closed.
    
    2. **Configuration via Environment**: Database credentials are injected
       via environment variables (.env file), not constructor parameters.
       This follows 12-factor app principles and keeps secrets out of code.
    
    3. **Consistent Return Types**: All mutation operations (upsert, update)
       return Dict[str, Any] with 'success' boolean and 'message' string.
       This provides consistent error handling across the API.
    
    4. **Embedded Queries**: Cypher queries are embedded in methods rather than
       in a separate queries.py file. This keeps related logic together and
       improves maintainability for domain-specific operations.
    
    5. **Transaction Support**: Use execute_transaction() for multi-query
       atomic operations. Individual methods use single queries for simplicity.
    
    Attributes:
        driver: Neo4j database driver instance
        uri: Database connection URI
        username: Database username
        password: Database password
    
    Example:
        >>> with ResearchGraph() as graph:
        ...     paper_data = {
        ...         'doi': '10.1101/2024.01.001',
        ...         'title': 'Novel Drug Discovery',
        ...         'abstract': 'This paper presents...',
        ...         'authors': ['Jane Doe', 'John Smith']
        ...     }
        ...     result = graph.upsert_paper(paper_data)
        ...     print(result['success'])
        True
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize the ResearchGraph connector.
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            username: Database username (defaults to NEO4J_USERNAME env var)
            password: Database password (defaults to NEO4J_PASSWORD env var)
        
        Raises:
            ValueError: If required credentials are not provided
            Exception: If connection to Neo4j fails
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not self.password:
            raise MissingCredentialsError(
                "NEO4J_PASSWORD must be set in environment or provided as parameter",
                context={"uri": self.uri, "username": self.username}
            )
        
        try:
            # Configure connection pooling for optimal performance
            self.driver: Driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=50,  # Max concurrent connections
                max_connection_lifetime=3600,  # 1 hour (in seconds)
                connection_acquisition_timeout=60.0  # 60 seconds timeout
            )
            logger.info(f"Connected to Neo4j at {self.uri} (pool size: 50, lifetime: 1h)")
            self._verify_connectivity()
        except AuthError as e:
            error_msg = f"Authentication failed for Neo4j at {self.uri}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(
                error_msg,
                context={"uri": self.uri, "username": self.username, "error": str(e)}
            ) from e
        except ServiceUnavailable as e:
            error_msg = f"Neo4j service unavailable at {self.uri}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(
                error_msg,
                context={"uri": self.uri, "error": str(e)}
            ) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to Neo4j at {self.uri}"
            logger.error(error_msg, exc_info=True)
            raise DatabaseError(error_msg, context={"error": str(e)}) from e
    
    def _verify_connectivity(self) -> None:
        """
        Verify connection to Neo4j database.
        
        Raises:
            Exception: If connection verification fails
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                if result.single()["num"] != 1:
                    raise DatabaseError(
                        "Neo4j connectivity test returned unexpected value",
                        context={"uri": self.uri}
                    )
            logger.info("Neo4j connectivity verified")
        except SessionExpired as e:
            error_msg = "Neo4j session expired during connectivity check"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(error_msg, context={"uri": self.uri}) from e
        except DatabaseError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            error_msg = f"Neo4j connectivity verification failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(error_msg, context={"uri": self.uri, "error": str(e)}) from e
    
    def close(self) -> None:
        """
        Close the database connection.
        
        This should be called when done with the graph connector to properly
        release resources.
        """
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a generic Cypher query and return results.
        
        Args:
            cypher_query: The Cypher query string to execute
            parameters: Optional dictionary of query parameters
        
        Returns:
            List of result records as dictionaries
        
        Example:
            >>> results = graph.query("MATCH (p:ResearchPaper) RETURN p.id AS id LIMIT 5")
            >>> for record in results:
            ...     print(record['id'])
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [dict(record) for record in result]
        except SessionExpired as e:
            error_msg = "Neo4j session expired during query execution"
            logger.error(f"{error_msg}: {cypher_query[:100]}", exc_info=True)
            raise QueryError(
                error_msg,
                context={"query_preview": cypher_query[:100], "error": str(e)}
            ) from e
        except Exception as e:
            error_msg = "Query execution failed"
            logger.error(f"{error_msg}: {cypher_query[:100]}", exc_info=True)
            raise QueryError(
                error_msg,
                context={"query_preview": cypher_query[:100], "parameters": str(parameters), "error": str(e)}
            ) from e
    
    def execute_transaction(
        self,
        queries: List[tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Execute multiple queries in a single transaction for atomicity.
        
        This ensures all operations succeed or all fail together, maintaining
        data consistency. Useful for complex multi-step operations like:
        - Creating a paper with multiple relationships
        - Updating multiple interconnected entities
        - Batch operations that must be atomic
        
        Args:
            queries: List of (query_string, parameters) tuples to execute
        
        Returns:
            Dictionary with execution results:
                - success (bool): Whether all queries succeeded
                - queries_executed (int): Number of queries executed
                - message (str): Status message
        
        Example:
            >>> queries = [
            ...     ("CREATE (p:Paper {doi: $doi})", {"doi": "10.1101/test"}),
            ...     ("CREATE (a:Author {name: $name})", {"name": "Dr. Smith"}),
            ...     ("MATCH (a:Author), (p:Paper) CREATE (a)-[:WROTE]->(p)", {})
            ... ]
            >>> result = graph.execute_transaction(queries)
        """
        def transaction_function(tx):
            results = []
            for query, params in queries:
                result = tx.run(query, params)
                results.append(result)
            return results
        
        try:
            with self.driver.session() as session:
                session.execute_write(transaction_function)
                logger.info(f"Transaction completed: {len(queries)} queries executed")
                return {
                    "success": True,
                    "queries_executed": len(queries),
                    "message": "Transaction completed successfully"
                }
        
        except Exception as e:
            logger.error(f"Transaction failed: {e}", exc_info=True)
            return {
                "success": False,
                "queries_executed": 0,
                "message": f"Transaction failed: {str(e)}"
            }
    
    def upsert_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert or update a research paper in the knowledge graph.
        
        This method performs the following operations:
        1. Creates or merges a Paper node (unique by DOI)
        2. Sets paper properties (title, abstract, date, etc.)
        3. Adds 'status: unprocessed' for workflow tracking
        4. Creates Author nodes for each author
        5. Creates :WROTE relationships from authors to paper
        
        Args:
            paper_data: Dictionary containing paper metadata with keys:
                - doi (str, required): Paper DOI (unique identifier)
                - title (str): Paper title
                - abstract (str): Paper abstract
                - authors (list): List of author names
                - category (str): Research category
                - date (str): Publication date
                - url (str): Full URL to paper
        
        Returns:
            Dictionary with operation results:
                - success (bool): Whether operation succeeded
                - paper_doi (str): DOI of the paper
                - authors_created (int): Number of author nodes created
                - message (str): Status message
        
        Raises:
            ValueError: If required fields (doi, title) are missing
            Exception: If database operation fails
        
        Example:
            >>> paper = {
            ...     'doi': '10.1101/2024.01.001',
            ...     'title': 'Drug Discovery',
            ...     'abstract': 'Abstract text...',
            ...     'authors': ['Dr. Smith', 'Dr. Jones'],
            ...     'category': 'pharmacology',
            ...     'date': '2024-01-15'
            ... }
            >>> result = graph.upsert_paper(paper)
            >>> print(result['authors_created'])
            2
        """
        # Validate required fields
        if not paper_data.get("doi"):
            raise ValidationError(
                "Paper DOI is required for upsert operation",
                context={"paper_data_keys": list(paper_data.keys())}
            )
        if not paper_data.get("title"):
            raise ValidationError(
                "Paper title is required for upsert operation",
                context={"doi": paper_data.get("doi"), "paper_data_keys": list(paper_data.keys())}
            )
        
        doi = paper_data["doi"]
        logger.info(f"Upserting paper: {doi}")
        
        # Cypher query to create/merge paper and authors
        query = """
        // Create or merge the Paper node with Pharmacology label for PubMed content
        MERGE (p:Paper:Pharmacology {doi: $doi})
        SET p.title = $title,
            p.abstract = $abstract,
            p.category = $category,
            p.date = $date,
            p.url = $url,
            p.status = 'unprocessed',
            p.updated_at = datetime()
        
        // Create Author nodes and relationships
        WITH p
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:WROTE]->(p)
        
        RETURN p.doi AS doi, 
               p.title AS title, 
               count(DISTINCT a) AS author_count
        """
        
        parameters = {
            "doi": doi,
            "title": paper_data.get("title", ""),
            "abstract": paper_data.get("abstract", ""),
            "category": paper_data.get("category", ""),
            "date": paper_data.get("date", ""),
            "url": paper_data.get("url", ""),
            "authors": paper_data.get("authors", [])
        }
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()
                
                if record:
                    author_count = record["author_count"]
                    logger.info(
                        f"Successfully upserted paper {doi} with {author_count} authors"
                    )
                    return {
                        "success": True,
                        "paper_doi": doi,
                        "authors_created": author_count,
                        "message": f"Paper {doi} upserted successfully"
                    }
                else:
                    logger.warning(f"No result returned for paper {doi}")
                    return {
                        "success": False,
                        "paper_doi": doi,
                        "authors_created": 0,
                        "message": "No result returned from database"
                    }
        
        except SessionExpired as e:
            error_msg = f"Session expired while upserting paper {doi}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "paper_doi": doi,
                "authors_created": 0,
                "message": error_msg,
                "error_type": "SessionExpired"
            }
        except Exception as e:
            error_msg = f"Failed to upsert paper {doi}: {type(e).__name__}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "paper_doi": doi,
                "authors_created": 0,
                "message": f"{error_msg}: {str(e)}",
                "error_type": type(e).__name__
            }
    
    def get_unprocessed_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve papers with 'unprocessed' status.
        
        This is useful for Phase 2 processing where papers need to be
        analyzed or enriched.
        
        Args:
            limit: Maximum number of papers to return
        
        Returns:
            List of paper dictionaries with doi, title, abstract, etc.
        """
        query = """
        MATCH (p:Paper {status: 'unprocessed'})
        RETURN p.doi AS doi,
               p.title AS title,
               p.abstract AS abstract,
               p.category AS category,
               p.date AS date
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"limit": limit})
                papers = [dict(record) for record in result]
                logger.info(f"Retrieved {len(papers)} unprocessed papers")
                return papers
        
        except Exception as e:
            logger.error(f"Failed to retrieve unprocessed papers: {e}")
            return []
    
    def update_paper_status(
        self,
        doi: str,
        status: str
    ) -> Dict[str, Any]:
        """
        Update the processing status of a paper.
        
        Args:
            doi: Paper DOI
            status: New status (e.g., 'processing', 'completed', 'failed')
        
        Returns:
            Dictionary with operation results:
                - success (bool): Whether update was successful
                - doi (str): Paper DOI
                - status (str): New status value
                - message (str): Status message
        """
        query = """
        MATCH (p:Paper {doi: $doi})
        SET p.status = $status,
            p.status_updated_at = datetime(),
            p.processed_at = CASE WHEN $status = 'processed' THEN datetime() ELSE p.processed_at END
        RETURN p.doi AS doi, p.status AS status
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {"doi": doi, "status": status})
                record = result.single()
                
                if record:
                    logger.info(f"Updated paper {doi} status to: {status}")
                    return {
                        "success": True,
                        "doi": record["doi"],
                        "status": record["status"],
                        "message": f"Status updated to '{status}'"
                    }
                else:
                    logger.warning(f"Paper {doi} not found for status update")
                    return {
                        "success": False,
                        "doi": doi,
                        "status": status,
                        "message": f"Paper {doi} not found"
                    }
        
        except Exception as e:
            logger.error(f"Failed to update paper {doi} status: {e}")
            return {
                "success": False,
                "doi": doi,
                "status": status,
                "message": f"Error: {str(e)}"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with counts of papers, authors, and relationships
        """
        query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (a:Author)
        OPTIONAL MATCH ()-[w:WROTE]->()
        RETURN count(DISTINCT p) AS paper_count,
               count(DISTINCT a) AS author_count,
               count(w) AS wrote_relationship_count
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                return {
                    "papers": record["paper_count"],
                    "authors": record["author_count"],
                    "relationships": record["wrote_relationship_count"]
                }
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"papers": 0, "authors": 0, "relationships": 0}
    
    def upsert_drug_interaction(
        self,
        drug_a: str,
        drug_b: str,
        interaction_type: str,
        evidence_text: str,
        source_doi: str,
        confidence: float = None,
        validation_status: str = None,
        chembl_validated: bool = False,
        drug_a_validated: bool = False,
        drug_b_validated: bool = False,
        conflict_detected: bool = False,
        potential_discovery: bool = False,
        needs_review: bool = False,
        requires_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Create or update a drug interaction in the knowledge graph.
        
        This method:
        1. Creates Drug nodes for both drugs (if they don't exist)
        2. Creates an :INTERACTS_WITH relationship between them
        3. Links the interaction to the source Paper
        4. Stores validation metadata from ChEMBL validation
        
        Args:
            drug_a: Name of the first drug
            drug_b: Name of the second drug
            interaction_type: Type of interaction (e.g., 'Inhibits', 'Potentiates')
            evidence_text: Supporting evidence from the paper
            source_doi: DOI of the source paper
            confidence: Optional confidence score (0.0 to 1.0)
            validation_status: ChEMBL validation status ('validated', 'partial', 'unknown', 'conflict')
            chembl_validated: Whether both drugs are validated in ChEMBL
            drug_a_validated: Whether drug A is validated in ChEMBL
            drug_b_validated: Whether drug B is validated in ChEMBL
            conflict_detected: Whether the interaction contradicts ChEMBL data
            potential_discovery: Whether this is a novel finding
            needs_review: Whether human review is required
            requires_validation: Whether further validation is needed
        
        Returns:
            Dictionary with operation results
        
        Example:
            >>> result = graph.upsert_drug_interaction(
            ...     drug_a="Warfarin",
            ...     drug_b="Aspirin",
            ...     interaction_type="Potentiates",
            ...     evidence_text="Aspirin enhances anticoagulant effect",
            ...     source_doi="10.1101/2024.01.001"
            ... )
        """
        logger.info(f"Upserting interaction: {drug_a} <-> {drug_b} ({interaction_type})")
        
        query = """
        // Create or merge Drug nodes
        MERGE (d1:Drug {name: $drug_a})
        ON CREATE SET d1.chembl_validated = $drug_a_validated
        ON MATCH SET d1.chembl_validated = COALESCE(d1.chembl_validated, $drug_a_validated)
        
        MERGE (d2:Drug {name: $drug_b})
        ON CREATE SET d2.chembl_validated = $drug_b_validated
        ON MATCH SET d2.chembl_validated = COALESCE(d2.chembl_validated, $drug_b_validated)
        
        // Create the interaction relationship using composite key properties
        // NOTE: Using separate properties instead of computed concatenation for better performance
        MERGE (d1)-[i:INTERACTS_WITH {
            drug_a: $drug_a,
            drug_b: $drug_b,
            source_doi: $source_doi
        }]-(d2)
        
        SET i.interaction_type = $interaction_type,
            i.evidence_text = $evidence_text,
            i.confidence = $confidence,
            i.validation_status = $validation_status,
            i.chembl_validated = $chembl_validated,
            i.conflict_detected = $conflict_detected,
            i.potential_discovery = $potential_discovery,
            i.needs_review = $needs_review,
            i.requires_validation = $requires_validation,
            i.updated_at = datetime()
        
        RETURN d1.name AS drug_a, 
               d2.name AS drug_b, 
               i.interaction_type AS interaction_type
        """
        
        parameters = {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "drug_a_validated": drug_a_validated,
            "drug_b_validated": drug_b_validated,
            "interaction_type": interaction_type,
            "evidence_text": evidence_text,
            "source_doi": source_doi,
            "confidence": confidence,
            "validation_status": validation_status,
            "chembl_validated": chembl_validated,
            "conflict_detected": conflict_detected,
            "potential_discovery": potential_discovery,
            "needs_review": needs_review,
            "requires_validation": requires_validation
        }
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()
                
                if record:
                    logger.info(
                        f"Successfully upserted interaction: "
                        f"{record['drug_a']} <-> {record['drug_b']}"
                    )
                    return {
                        "success": True,
                        "drug_a": record["drug_a"],
                        "drug_b": record["drug_b"],
                        "interaction_type": record["interaction_type"],
                        "message": "Interaction upserted successfully"
                    }
                else:
                    logger.warning("No result returned from drug interaction upsert")
                    return {
                        "success": False,
                        "message": "No result returned from database"
                    }
        
        except Exception as e:
            logger.error(
                f"Failed to upsert interaction {drug_a} <-> {drug_b}: {e}",
                exc_info=True
            )
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    def get_drug_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about drugs and interactions in the database.
        
        Returns:
            Dictionary with drug and interaction counts
        """
        query = """
        MATCH (d:Drug)
        OPTIONAL MATCH ()-[i:INTERACTS_WITH]-()
        RETURN count(DISTINCT d) AS drug_count,
               count(DISTINCT i) AS interaction_count
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                return {
                    "drugs": record["drug_count"],
                    "interactions": record["interaction_count"]
                }
        
        except Exception as e:
            logger.error(f"Failed to get drug statistics: {e}")
            return {"drugs": 0, "interactions": 0}
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with all counts (papers, authors, drugs, interactions)
        """
        query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (a:Author)
        OPTIONAL MATCH (d:Drug)
        OPTIONAL MATCH ()-[w:WROTE]->()
        OPTIONAL MATCH ()-[i:INTERACTS_WITH]-()
        RETURN count(DISTINCT p) AS paper_count,
               count(DISTINCT a) AS author_count,
               count(DISTINCT d) AS drug_count,
               count(w) AS wrote_count,
               count(DISTINCT i) AS interaction_count
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                record = result.single()
                return {
                    "papers": record["paper_count"],
                    "authors": record["author_count"],
                    "drugs": record["drug_count"],
                    "relationships": {
                        "wrote": record["wrote_count"],
                        "interactions": record["interaction_count"]
                    }
                }
        
        except Exception as e:
            logger.error(f"Failed to get comprehensive statistics: {e}")
            return {
                "papers": 0,
                "authors": 0,
                "drugs": 0,
                "relationships": {"wrote": 0, "interactions": 0}
            }

