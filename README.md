# Aura Discovery

## Agentic Graph-RAG System for Biomedical Discovery

A sophisticated biomedical knowledge discovery platform that combines Google Gemini, LangGraph, and Neo4j to construct and query intelligent knowledge graphs from scientific literature.

## 🚀 Features

- **Agentic Workflow**: Multi-agent system using LangGraph for coordinated knowledge extraction
- **Graph-RAG**: Retrieval-Augmented Generation enhanced with knowledge graph structure
- **Gemini Integration**: Powered by Google's Gemini 1.5 Pro for advanced reasoning
- **Neo4j Graph Database**: Efficient storage and querying of biomedical relationships
- **External API Integration**: Automated enrichment from PubMed and ChEMBL databases

## 📁 Project Structure

```
Aura-Discovery/
├── data/
│   └── raw/              # Initial PubMed JSONs
├── src/
│   ├── agents/           # LangGraph node logic
│   │   ├── researcher.py
│   │   └── validator.py
│   ├── database/         # Neo4j connections and Cypher queries
│   │   ├── neo4j_manager.py
│   │   └── queries.py
│   ├── tools/            # API integrations
│   │   ├── pubmed_api.py
│   │   └── chembl_api.py
│   ├── state.py          # TypedDict shared state
│   └── factory.py        # Pre-configured Gemini models
├── .env.example          # Environment variable template
├── .gitignore
├── pyproject.toml        # uv dependency management
└── README.md
```

## 🛠️ Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Neo4j database (local or cloud)
- Google AI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Aura-Discovery
   ```

2. **Install dependencies**
   
   **Option A: Using uv (recommended)**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # On Windows: irm https://astral.sh/uv/install.ps1 | iex
   
   # Create virtual environment and install dependencies
   uv venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   uv pip install -e .
   ```
   
   **Option B: Using pip**
   ```bash
   # Create virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Or install in development mode
   pip install -e .
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your credentials:
   # - GOOGLE_API_KEY
   # - NEO4J_URI
   # - NEO4J_PASSWORD
   ```

4. **Verify Neo4j connection**
   ```python
   from src.database.graph_connector import ResearchGraph
   
   graph = ResearchGraph()
   print(graph.get_statistics())
   graph.close()
   ```

## 🚦 Quick Start

### Phase 1: Ingest Research Papers

Run the main ingestion script to fetch and store papers from PubMed:

```bash
python ingest_research.py
```

This will:
- Fetch the latest 10 papers from the 'bioinformatics' category
- Store them in Neo4j with `status: 'unprocessed'`
- Create author nodes and relationships
- Generate detailed logs in `ingest_research.log`

**Customizing the ingestion:**

```python
from ingest_research import ingest_papers

# Ingest 20 papers from bioinformatics
stats = ingest_papers(category="bioinformatics", count=20)
print(f"Ingested {stats['total_ingested']} papers")
```

### Phase 2: Agentic Validation Pipeline

Run the agentic validation workflow to extract, validate, and flag drug interactions:

```bash
python process_with_validation.py
```

This will:
- Extract drug interactions using Gemini 2.5-flash with structured output
- **Validate** each interaction against the ChEMBL database
- **Detect conflicts** (paper contradicts known pharmacology)
- **Identify discoveries** (novel drugs/interactions not in ChEMBL)
- Create Drug nodes and INTERACTS_WITH relationships with validation metadata
- Flag conflicts with red nodes and discoveries with gold nodes
- Generate detailed logs showing validation results

**The Agentic Workflow:**

```
┌────────────┐
│  Extract   │  Use Gemini to get DDIs from abstract
└─────┬──────┘
      │
      ▼
┌────────────┐
│  Validate  │  Check each drug against ChEMBL
└─────┬──────┘
      │
      ├──► Conflicts detected? ──► Flag Conflict (red nodes)
      │                                    │
      ├──► Novel discoveries? ──► Mark Discovery (gold nodes)
      │                                    │
      └──► All validated? ─────────────────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │Update Graph │
                                    └─────────────┘
```

**Validation Categories:**

1. **Validated** 🟢: Both drugs found in ChEMBL, interaction consistent
2. **Conflict** 🔴: Paper contradicts known ChEMBL data (e.g., "toxic" vs "safe")
3. **Discovery** 🟡: Unknown drugs or novel interactions not in ChEMBL
4. **Partial** 🟠: One drug found, one unknown

**Results:**
```python
from src.database.graph_connector import ResearchGraph

with ResearchGraph() as graph:
    # Query validated interactions
    validated = graph.query("""
        MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
        WHERE r.chembl_validated = true
        RETURN d1.name, d2.name, r.interaction_type, r.validation_status
    """)
    
    # Query conflicts
    conflicts = graph.query("""
        MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
        WHERE r.conflict_detected = true
        RETURN d1.name, d2.name, r.interaction_type
    """)
    
    # Query discoveries
    discoveries = graph.query("""
        MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
        WHERE r.potential_discovery = true
        RETURN d1.name, d2.name, r.interaction_type
    """)
```

**Direct Usage:**

```python
from src.agents.discovery_agent import discover_and_validate

# Process a single paper
result = discover_and_validate(
    paper_id="PMID:12345678",
    abstract="Warfarin and aspirin show synergistic anticoagulant effects...",
    title="Drug Interactions in Cardiology"
)

print(f"Extracted: {len(result['extracted_ddis'])} interactions")
print(f"Validated: {result['graph_update_results']['validated']}")
print(f"Conflicts: {result['graph_update_results']['conflicts']}")
print(f"Discoveries: {result['graph_update_results']['discoveries']}")
```

See [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) for detailed documentation.

```python
from ingest_research import ingest_papers

# Ingest 20 papers from bioinformatics
stats = ingest_papers(category="bioinformatics", count=20)
print(f"Ingested {stats['total_ingested']} papers")
```

### General Usage

```python
from src.factory import default_llm, structured_llm
from src.state import AgentState
from src.tools.biorxiv_api import search_biorxiv

# Search for papers
papers = search_biorxiv("cancer immunotherapy", max_results=10)

# Use Gemini for analysis
response = default_llm.invoke("Summarize key findings in cancer immunotherapy")
print(response.content)

# Initialize agent state
state: AgentState = {
    "query": "cancer immunotherapy",
    "papers": papers,
    "entities": [],
    "relationships": [],
    "messages": []
}
```

## 🏗️ Architecture

### Agent Workflow

1. **Researcher Agent**: Searches PubMed for relevant papers
2. **Extractor Agent**: Extracts entities and relationships using Gemini
3. **Validator Agent**: Validates and scores extracted information
4. **Enricher Agent**: Augments data from ChEMBL and other sources
5. **Graph Builder**: Constructs Neo4j knowledge graph
6. **Query Agent**: Answers user questions using Graph-RAG

### State Management

All agents share a typed state object (see `src/state.py`) that flows through the LangGraph workflow.

## 🤖 Using Gemini Models

```python
from src.factory import get_gemini_llm, structured_llm, reasoning_llm

# Default model (balanced)
llm = get_gemini_llm()

# For structured extraction (lower temperature)
extractor = structured_llm

# For creative reasoning (higher temperature)
reasoner = reasoning_llm

# Custom configuration
custom_llm = get_gemini_llm(
    temperature=0.5,
    max_output_tokens=4096
)
```

## 📊 Neo4j Schema

- **Nodes**: Entity, Gene, Protein, Compound, Paper
- **Relationships**: INTERACTS_WITH, RELATED_TO, MENTIONED_IN
- All relationships include confidence scores and evidence links

## 🔧 Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key | Required |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `LOG_LEVEL` | Logging level | `INFO` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Google Gemini for advanced language understanding
- LangGraph for agent orchestration
- Neo4j for graph database capabilities
- PubMed and ChEMBL for open biomedical data

## 📧 Contact

[Your contact information]

---

Built with ❤️ for biomedical discovery
