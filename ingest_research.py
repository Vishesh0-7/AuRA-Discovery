"""
Research Paper Ingestion Script - Phase 1

This script is the main entry point for ingesting research papers from PubMed
into the Neo4j knowledge graph. It fetches the latest papers and stores them
with an 'unprocessed' status for subsequent analysis phases.

Usage:
    python ingest_research.py

Environment Variables Required:
    - NEO4J_URI: Neo4j database URI
    - NEO4J_USERNAME: Neo4j username
    - NEO4J_PASSWORD: Neo4j password

Author: Aura Discovery Team
Date: March 2026
"""

import logging
import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.tools.pubmed_api import get_latest_papers
from src.database.graph_connector import ResearchGraph

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_research.log", encoding="utf-8")
    ]
)

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logger = logging.getLogger(__name__)


def chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def ingest_papers(
    query: str = "drug interaction[MeSH Terms] AND pharmacology[MeSH Terms]",
    count: int = 50
) -> Dict[str, Any]:
    """
    Fetch and ingest research papers into the Neo4j graph database.
    
    This function orchestrates the complete ingestion pipeline:
    1. Fetches latest papers from PubMed
    2. Validates paper data
    3. Upserts papers into Neo4j with 'unprocessed' status
    4. Creates author nodes and relationships
    
    Args:
        query: PubMed search query (MeSH terms recommended)
               Default: 'drug interaction[MeSH Terms] AND pharmacology[MeSH Terms]'
        count: Number of papers to fetch (default: 50)
    
    Returns:
        Dictionary with ingestion statistics:
            - total_fetched: Number of papers fetched from API
            - total_ingested: Number of papers successfully ingested
            - failed: Number of papers that failed to ingest
            - errors: List of error messages
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Research Paper Ingestion")
    logger.info("=" * 70)
    logger.info(f"Query: {query}")
    logger.info(f"Target count: {count}")
    
    # Statistics tracking
    stats = {
        "total_fetched": 0,
        "total_ingested": 0,
        "failed": 0,
        "errors": []
    }
    
    try:
        # Step 1: Fetch papers from PubMed
        logger.info("\n[Step 1/3] Fetching papers from PubMed...")
        papers = get_latest_papers(query=query, count=count)
        stats["total_fetched"] = len(papers)
        
        if not papers:
            logger.warning("No papers retrieved from PubMed")
            return stats
        
        logger.info(f"[OK] Successfully fetched {len(papers)} papers")
        
        # Step 2: Connect to Neo4j
        logger.info("\n[Step 2/3] Connecting to Neo4j database...")
        with ResearchGraph() as graph:
            logger.info("[OK] Connected to Neo4j successfully")
            
            # Get initial statistics
            initial_stats = graph.get_statistics()
            logger.info(
                f"Database stats: {initial_stats['papers']} papers, "
                f"{initial_stats['authors']} authors"
            )
            
            # Step 3: Ingest papers
            logger.info(f"\n[Step 3/3] Ingesting {len(papers)} papers into graph in batches...")

            batch_size = 50
            paper_batches = chunked(papers, batch_size)

            for batch_index, paper_batch in enumerate(paper_batches, 1):
                logger.info(
                    "  [batch %d/%d] Upserting %d papers",
                    batch_index,
                    len(paper_batches),
                    len(paper_batch),
                )

                result = graph.batch_upsert_papers(paper_batch)

                if result["success"]:
                    stats["total_ingested"] += result.get("processed", 0)
                    stats["failed"] += result.get("failed", 0)
                    for skipped_item in result.get("skipped", []):
                        stats["errors"].append(f"Skipped paper: {skipped_item}")
                    logger.info(
                        "    [OK] Batch stored %d papers",
                        result.get("processed", 0),
                    )
                else:
                    stats["failed"] += len(paper_batch)
                    stats["errors"].append(result["message"])
                    logger.error("    [FAIL] %s", result["message"])

                    # Fallback to row-by-row upserts for the failed batch so we keep partial progress.
                    for idx, paper in enumerate(paper_batch, 1):
                        try:
                            if not paper.get("doi"):
                                logger.warning("Paper %d in failed batch: Missing DOI, skipping", idx)
                                stats["failed"] += 1
                                continue

                            fallback = graph.upsert_paper(paper)
                            if fallback["success"]:
                                stats["total_ingested"] += 1
                            else:
                                stats["failed"] += 1
                                stats["errors"].append(fallback["message"])
                        except Exception as exc:
                            stats["failed"] += 1
                            stats["errors"].append(str(exc))
            
            # Get final statistics
            final_stats = graph.get_statistics()
            papers_added = final_stats["papers"] - initial_stats["papers"]
            authors_added = final_stats["authors"] - initial_stats["authors"]
            
            logger.info("\n" + "=" * 70)
            logger.info("INGESTION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Papers fetched:     {stats['total_fetched']}")
            logger.info(f"Papers ingested:    {stats['total_ingested']}")
            logger.info(f"Papers failed:      {stats['failed']}")
            logger.info(f"New papers added:   {papers_added}")
            logger.info(f"New authors added:  {authors_added}")
            logger.info(f"Total in database:  {final_stats['papers']} papers, "
                       f"{final_stats['authors']} authors")
            
            if stats["errors"]:
                logger.warning(f"\nEncountered {len(stats['errors'])} errors:")
                for error in stats["errors"][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")
                if len(stats["errors"]) > 5:
                    logger.warning(f"  ... and {len(stats['errors']) - 5} more")
    
    except Exception as e:
        logger.error(f"Critical error during ingestion: {e}", exc_info=True)
        stats["errors"].append(f"Critical error: {str(e)}")
        raise
    
    return stats


def main():
    """
    Main entry point for the ingestion script.
    
    Loads environment variables and executes the ingestion pipeline
    with default parameters.
    """
    try:
        # Load environment variables from .env file
        logger.info("Loading environment variables...")
        load_dotenv()
        logger.info("[OK] Environment variables loaded")

        target_count = int(os.getenv("PUBMED_TARGET_COUNT", "2000"))
        logger.info("Using PUBMED_TARGET_COUNT=%d", target_count)
        
        # Execute ingestion with a broader DDI query for higher recall.
        stats = ingest_papers(
            query='("drug interaction"[Title/Abstract] OR "drug-drug interaction"[Title/Abstract] OR DDI[Title/Abstract] OR "pharmacokinetic interaction"[Title/Abstract] OR "pharmacodynamic interaction"[Title/Abstract])',
            count=target_count
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Successfully ingested {stats['total_ingested']} papers into the graph.")
        
        if stats["failed"] > 0:
            print(f"[WARNING] {stats['failed']} papers failed to ingest")
            print("Check ingest_research.log for details")
            sys.exit(1)
        else:
            print("[OK] All papers ingested successfully!")
            sys.exit(0)
    
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"\n[FATAL] Fatal error: {e}")
        print("Check ingest_research.log for full traceback")
        sys.exit(1)


if __name__ == "__main__":
    main()
