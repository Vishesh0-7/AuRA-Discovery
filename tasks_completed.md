# Tasks Completed: Aura-Discovery Project

This document summarizes the end-to-end work completed in the project, with emphasis on methodology, modeling stages, evaluation, and reproducible artifacts for research writing.

## 1. Data Ingestion and Corpus Expansion

### 1.1 Literature and Data Sources Integrated
- PubMed ingestion pipeline implemented and used for biomedical paper retrieval.
- BioRxiv API tooling integrated.
- ChEMBL API tooling integrated for validation/reference support.
- Neo4j graph database integrated as the project knowledge backbone.

### 1.2 Historical Coverage Control
- PubMed lookback window made configurable via `PUBMED_YEARS_BACK`.
- Default historical window preserved with option to expand retrieval further back in time.

### 1.3 Graph Persistence
- Interaction records stored as Neo4j relationships (`INTERACTS_WITH`) with metadata such as validation status and evidence text.

## 2. Dataset Construction and Labeling

### 2.1 DDI Dataset Build
- Structured dataset generation script implemented: `build_ddi_dataset.py`.
- Output dataset schema finalized:
  - `drug1`
  - `drug2`
  - `text`
  - `label`

### 2.2 Label Definition
- Binary target labeling standardized:
  - `label = 1` when `validation_status == 'validated'`
  - `label = 0` otherwise

### 2.3 Data Quality and Imbalance Analysis (EDA)
- Class distribution measured and logged.
- Positive/negative imbalance explicitly quantified.
- EDA visual artifacts generated under `reports/figures`.

## 3. Imbalance Handling and Balanced Dataset Creation

### 3.1 Negative Sample Generation
- Synthetic negative pairs generated from random drug combinations not present in positive interaction pairs.
- Configurable negative-to-positive ratio support (1:1 and 1:2 modes).

### 3.2 Balanced Dataset Output
- Balanced dataset saved to:
  - `data/processed/ddi_dataset_balanced.csv`
- Raw and balanced distributions both persisted and documented.

### 3.3 EDA Visual Deliverables
- Class distribution before/after balancing plot.
- Drug frequency distribution plot.
- Additional exploratory charts retained in `reports/figures`.

## 4. Feature Engineering Pipeline

### 4.1 Text Features
- TF-IDF features on evidence/abstract text implemented.
- Configurable `max_features`, n-grams, and normalization settings.

### 4.2 Drug-Pair Features
- Pair identifier encoding features added.
- Numeric pair features added (e.g., lexical overlap and pair-level heuristics).

### 4.3 Numeric Normalization
- Numeric pair features normalized using `StandardScaler`.

### 4.4 ML-Ready Data Split
- Reusable train/test preparation implemented via `load_ml_ready_split` in `src/features/pipeline.py`.
- Returns `X_train`, `X_test`, `y_train`, `y_test` ready for model training.

## 5. Traditional ML Modeling (Text + Pair Features)

### 5.1 Baseline and Improved Training Pipelines
- Baseline training script implemented and iteratively improved:
  - `train_baseline_models.py`
- Dedicated traditional model trainer added:
  - `train_traditional_models.py`

### 5.2 Models Trained
- Logistic Regression variants.
- XGBoost variants.

### 5.3 Imbalance Corrections Applied
- `class_weight='balanced'` for Logistic Regression.
- `scale_pos_weight` for XGBoost.
- SMOTE experiments performed and compared.

### 5.4 Evaluation Metrics Standardized
- Precision
- Recall
- F1-score
- PR-AUC
- (Additional diagnostics in some runs: confusion matrix, macro/balanced metrics)

### 5.5 Model Artifacts Saved
- Models and feature pipelines persisted in `artifacts/models`.
- Metrics reports persisted in `artifacts/metrics`.

## 6. LLM Model (Model A)

### 6.1 Structured Prediction Schema
- LLM output schema created for DDI prediction:
  - Initial: `DDIPrediction`
  - Stage-4 strict schema: `Stage4LLMPrediction`

### 6.2 Prediction API
- Implemented in `src/agents/predictor.py`:
  - `predict_llm(...)`
  - `predict_llm_safe(...)`
  - `predict_llm_stage4(...)`
  - `predict_llm_stage4_safe(...)`

### 6.3 Local LLM Runtime
- Ollama-based local inference configured (Llama 3.1 8B).
- Structured output enforcement used for strict JSON predictions.

### 6.4 Batch LLM Evaluation Outputs
- Batch prediction runner implemented:
  - `run_stage4_llm_predictions.py`
- Stored outputs:
  - `artifacts/predictions/stage4_llm_predictions.jsonl`
  - `artifacts/predictions/stage4_llm_summary.json`

## 7. Graph Modeling (Model C)

### 7.1 Initial Graph Feature Set
- Common neighbors
- Degree-based features
- Jaccard similarity
- Support count features

### 7.2 Extended Graph Feature Engineering
- Adamic-Adar
- Resource allocation
- Salton
- Sorensen
- Hub-promoted / hub-depressed
- Degree centrality
- Path existence

### 7.3 Improved Graph Signal Expansion
- Additional topology features implemented:
  - PageRank
  - Shortest path length (and inverse)
  - Shared-neighbor explicit feature
- Optional Node2Vec-derived features integrated:
  - cosine similarity
  - L2 distance
- Graceful fallback when Node2Vec dependencies are unavailable.

### 7.4 Graph-Only Training Scripts
- `train_graph_model.py` (baseline/improved topology set)
- `train_graph_model_improved.py` (enhanced signal set + baseline comparison)

### 7.5 Graph Artifacts and Reports
- Graph model artifacts saved under `artifacts/models`.
- Metrics and baseline-vs-improved deltas saved under:
  - `artifacts/metrics/graph_metrics.json`
  - `artifacts/metrics/graph_metrics_improved.json`

## 8. Hybrid Modeling (Model D)

### 8.1 Early Fusion
- Combined feature matrix built from:
  - TF-IDF text features
  - Graph features
  - LLM confidence features
- XGBoost trained on fused features.

### 8.2 Late Fusion
- Separate model outputs combined using fixed weighted rule:
  - `final_score = 0.5 * ML + 0.3 * Graph + 0.2 * LLM`

### 8.3 Hybrid Training Implementations
- `train_hybrid_model.py` (iterative hybrid pipeline)
- `train_hybrid_fusion_pipeline.py` (stage-based hybrid workflow)
- `train_stage6_hybrid.py` (explicit early-vs-late fusion comparison)

### 8.4 LLM Budget and Caching
- LLM-call budget controls implemented.
- Cache-backed pair prediction storage enabled:
  - `artifacts/features/llm_pair_predictions.json`

### 8.5 Hybrid Outputs
- Hybrid models persisted in `artifacts/models`.
- Hybrid metrics persisted in `artifacts/metrics` and `reports`.

## 9. Full Evaluation and Comparative Analysis

### 9.1 Unified Stage-7 Evaluation
- Script implemented:
  - `stage7_full_evaluation.py`

### 9.2 Models Compared
- Logistic Regression
- XGBoost
- Graph model
- LLM model
- Hybrid model

### 9.3 Outputs Produced
- Comparison table (CSV): `reports/stage7_model_comparison.csv`
- Comparison table (Markdown): `reports/stage7_model_comparison.md`
- Full summary JSON: `reports/stage7_full_evaluation.json`
- Comparison bar chart: `reports/figures/stage7_model_comparison_bars.png`
- Imbalance-effect chart: `reports/figures/stage7_imbalance_effect.png`

### 9.4 Imbalance Handling Insights Captured
- Raw skew vs balanced distributions documented.
- Class-weight and scale-pos-weight behavior tracked.
- SMOTE sensitivity outcomes explicitly reported.

## 10. Reproducibility and Experiment Tracking

### 10.1 Artifact Strategy
- Model binaries stored in `artifacts/models`.
- Metrics snapshots/versioned outputs in `artifacts/metrics`.
- Human-readable reports and figures in `reports`.

### 10.2 Determinism Controls
- Random seeds used across major training scripts.
- Fixed splits and thresholds recorded in metrics outputs.

### 10.3 Methodology Readiness for Paper Writing
This project now includes all major components typically needed for the methodology and experiments sections of a research paper:
- data acquisition and labeling protocol
- imbalance correction strategy
- feature engineering design
- model families and training setups
- evaluation metrics and comparative analysis
- ablation-like comparisons (e.g., graph baseline vs improved; early vs late fusion)
- persisted artifacts for reproducibility

## 11. Key Files Implemented/Updated During Project

- `build_ddi_dataset.py`
- `src/features/pipeline.py`
- `src/features/graph_pipeline.py`
- `src/schema.py`
- `src/agents/predictor.py`
- `train_baseline_models.py`
- `train_traditional_models.py`
- `train_graph_model.py`
- `train_graph_model_improved.py`
- `train_hybrid_model.py`
- `train_hybrid_fusion_pipeline.py`
- `train_stage6_hybrid.py`
- `run_stage4_llm_predictions.py`
- `stage7_full_evaluation.py`

## 12. Suggested Citation-Oriented Sections You Can Draft Next

- Problem formulation and task definition (binary DDI prediction)
- Data collection and label policy
- Graph construction and feature extraction
- Model architectures (Traditional, LLM, Graph, Hybrid)
- Experimental setup (splits, imbalance handling, metrics)
- Comparative results and error analysis
- Limitations (e.g., LLM fallback coverage, Node2Vec dependency availability)
- Future work (calibration, external validation, real-world deployment)
