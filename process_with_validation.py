"""
Process Literature with Agentic Validation

This script runs the complete Discovery Agent workflow on papers
stored in Neo4j, validating extracted interactions against ChEMBL.

Features:
- Extracts DDIs from abstracts using Ollama
- Validates against ChEMBL database
- Detects conflicts and novel discoveries
- Updates Neo4j with appropriate labels

Configuration:
- Processes all unprocessed papers per run
- Rate limited: 1 second between papers for local execution
"""

import logging
import os
import sys
import time
from typing import List, Dict, Any
from src.agents.discovery_agent import DiscoveryAgent
from src.database.graph_connector import ResearchGraph

# Configure logging with file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("process_validation.log", encoding="utf-8")
    ]
)

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


def get_unprocessed_papers(graph: ResearchGraph) -> List[Dict[str, Any]]:
    """
    Fetch all papers from Neo4j that haven't been validated yet.
    
    Pulls papers with unprocessed/pending/null status directly from Neo4j.
    
    Args:
        graph: ResearchGraph instance
    
    Returns:
        List of paper dictionaries with id, title, abstract
    """
    # Fetch all unprocessed papers in one run for local processing.
    query = """
    MATCH (p:Paper)
    WHERE p.status IN ['unprocessed', 'pending'] OR p.status IS NULL
    RETURN p.doi AS doi,
           p.title AS title,
           p.abstract AS abstract
    ORDER BY p.date DESC
    """
    papers = graph.query(query)
    
    # Convert field names to match expected format
    formatted_papers = []
    for paper in papers:
        formatted_papers.append({
            'paper_id': paper['doi'],
            'title': paper['title'],
            'abstract': paper['abstract']
        })
    
    logger.info(f"Found {len(formatted_papers)} unprocessed papers")
    return formatted_papers


def get_all_papers(graph: ResearchGraph, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch papers from Neo4j for processing.
    
    Args:
        graph: ResearchGraph instance
        limit: Maximum number of papers to process (default: 5)
    
    Returns:
        List of paper dictionaries with id, title, abstract
    """
    query = """
    MATCH (p:Paper)
    RETURN p.doi AS paper_id, p.title AS title, p.abstract AS abstract
    ORDER BY p.date DESC
    LIMIT $limit
    """
    
    results = graph.query(query, {'limit': limit})
    papers = [dict(record) for record in results]
    logger.info(f"Found {len(papers)} papers (limit: {limit})")
    return papers


def main():
    """
    Main processing pipeline with agentic validation.
    """
    logger.info("=" * 80)
    logger.info("Starting agentic validation pipeline")
    logger.info("Logging to: process_validation.log")
    logger.info("=" * 80)
    
    # Initialize components
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    agent = DiscoveryAgent(model_name=model_name)
    logger.info(f"Using Ollama model: {model_name}")
    graph = ResearchGraph()
    
    try:
        # Get papers to process - only fetch unprocessed ones to avoid duplicates
        papers = get_unprocessed_papers(graph)
        
        if not papers:
            logger.warning("No unprocessed papers found in database. All papers have been validated.")
            return
        
        logger.info(f"Processing {len(papers)} papers with validation workflow")
        logger.info("=" * 80)
        
        # Statistics
        stats = {
            'total': len(papers),
            'processed': 0,
            'interactions_found': 0,
            'validated': 0,
            'conflicts': 0,
            'discoveries': 0,
            'errors': 0
        }
        
        # Process each paper
        for i, paper in enumerate(papers, 1):
            paper_id = paper['paper_id']
            abstract = paper.get('abstract', '')
            title = paper.get('title', '')
            
            if not abstract:
                logger.warning(f"[{i}/{len(papers)}] No abstract for {paper_id}, skipping")
                continue
            
            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper_id}")
            logger.info(f"Title: {title[:100]}...")
            
            try:
                # Run the discovery agent workflow
                result = agent.process_paper(
                    paper_id=paper_id,
                    abstract=abstract,
                    title=title
                )
                
                # Update statistics
                stats['processed'] += 1
                stats['interactions_found'] += len(result.get('extracted_ddis', []))
                stats['validated'] += result.get('graph_update_results', {}).get('validated', 0)
                stats['conflicts'] += len(result.get('conflicts', []))
                stats['discoveries'] += len(result.get('discoveries', []))
                
                if result.get('error'):
                    stats['errors'] += 1
                    logger.error(f"  ⚠️ ERROR in workflow: {result.get('error')}")
                    logger.error(f"  - Paper may be partially processed")
                
                # Log results
                logger.info(f"  - Extracted: {len(result.get('extracted_ddis', []))} interactions")
                logger.info(f"  - Validated: {result.get('graph_update_results', {}).get('validated', 0)}")
                logger.info(f"  - Conflicts: {len(result.get('conflicts', []))}")
                logger.info(f"  - Discoveries: {len(result.get('discoveries', []))}")
                logger.info(f"  - Decision: {result.get('final_decision', 'unknown')}")
                
                # Mark paper as processed (if no error occurred)
                if not result.get('error'):
                    try:
                        # Use the proper update_paper_status method
                        update_result = graph.update_paper_status(paper_id, 'processed')
                        if update_result['success']:
                            logger.info(f"  - Marked paper as processed")
                        else:
                            logger.warning(f"  - Failed to mark paper as processed: {update_result.get('message')}")
                    except Exception as e:
                        logger.warning(f"  - Failed to update paper status: {e}")
                
                # Brief pause between papers to avoid overwhelming local resources.
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"[{i}/{len(papers)}] Error processing {paper_id}: {e}", exc_info=True)
                stats['errors'] += 1
                time.sleep(5)
        
        # Final report
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total papers: {stats['total']}")
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Interactions found: {stats['interactions_found']}")
        logger.info(f"Validated: {stats['validated']}")
        logger.info(f"Conflicts detected: {stats['conflicts']}")
        logger.info(f"Potential discoveries: {stats['discoveries']}")
        logger.info(f"Errors: {stats['errors']}")
        if stats['errors'] > 0:
            logger.warning(f"⚠️ {stats['errors']} error(s) occurred. Check process_validation.log for details.")
        logger.info("=" * 80)
        logger.info("Full log saved to: process_validation.log")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    
    finally:
        graph.close()
        logger.info("Pipeline shutdown complete")


if __name__ == "__main__":
    main()
