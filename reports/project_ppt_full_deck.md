# Aura-Discovery: End-to-End DDI Prediction (PPT Content)

## Slide 1 - Project Introduction and Problem Statement

**Title:** Aura-Discovery: Drug-Drug Interaction (DDI) Prediction for Pharmacovigilance

**Problem Context**
- Adverse drug-drug interactions are a major medication safety risk.
- Missing a true DDI (false negative) can lead to severe clinical consequences.
- Existing approaches often optimize overall score but may not prioritize safety-critical recall.

**Objective**
- Build and compare multiple DDI prediction strategies using text, graph topology, and LLM signals.
- Prioritize a practical trade-off: high predictive quality with strong safety behavior.
- Deliver a deployable scoring pipeline for pair-level DDI screening.

**One-Line Aim**
- Develop a robust DDI prediction system that balances performance and clinical safety, then identify the best model family for deployment.

---

## Slide 2 - New EDA: Dataset Composition and Imbalance

**Title:** EDA 1/3 - Dataset Structure and Label Distribution

**Dataset Facts (from data/processed/ddi_dataset.csv)**
- Total rows: 1,365
- Positive labels: 1,272 (93.19%)
- Negative labels: 93 (6.81%)
- Positive:Negative ratio: 13.68:1
- Unique drugs: 1,066
- Unique canonical drug pairs: 1,275

**Insight**
- The dataset is highly imbalanced toward positive DDIs.
- This makes accuracy alone misleading and motivates PR-AUC, recall, macro-F1, and threshold tuning.

**What to Visualize**
- Pie or bar chart: positive vs negative classes.
- Card metrics: rows, unique drugs, unique pairs.

---

## Slide 3 - New EDA: Text Evidence Characteristics

**Title:** EDA 2/3 - Evidence Text Profile

**Text Statistics**
- Mean text length: 117.16 characters
- Median text length: 107 characters
- Mean token length: 15.93 words
- Median token length: 14 words
- 90th percentile token length: 27 words

**Insight**
- Evidence snippets are short-to-medium, suitable for TF-IDF features.
- Short abstracts/evidence fragments justify combining text with graph signals for context completion.

**What to Visualize**
- Histogram: text word counts.
- Boxplot: text length spread.

---

## Slide 4 - New EDA: Frequency and Coverage Patterns

**Title:** EDA 3/3 - Drug/Pair Frequency Patterns

**Observed Patterns**
- Repeated canonical pairs (>1 record): 79
- Most frequent drugs include:
  - rifampicin (32), itraconazole (31), warfarin (31), tacrolimus (30), ritonavir (26)
- Most frequent pairs include:
  - glasmacinal + verapamil (4)
  - amiodarone + apixaban (4)
  - several pairs with frequency 3

**Insight**
- Frequency concentration suggests potential bias toward common drugs.
- Group-aware splitting by canonical pair is necessary to reduce pair leakage across splits.

**What to Visualize**
- Bar chart: top-10 drug frequency.
- Bar chart: top-10 pair frequency.

---

## Slide 5 - Proposed Solution: Methodology Overview

**Title:** Proposed Solution 1/4 - Multi-Model Methodology

**Compared Modeling Streams**
- Traditional text ML:
  - Logistic Regression (class-weight and SMOTE variants)
  - XGBoost (weighted and SMOTE variants)
- Graph-feature models:
  - Engineered topology features (degree, common neighbors, jaccard, adamic-adar, etc.)
- Graph Neural Network:
  - GraphSAGE edge classifier with message passing
- LLM confidence model:
  - LLM-derived confidence as an additional predictive signal
- Hybrid fusion:
  - Late fusion of text + graph + LLM channels

**Why this design**
- No single modality is guaranteed to capture all interaction evidence.
- Ensemble-style comparison exposes best safety/performance profile.

---

## Slide 6 - Proposed Solution: Required System Architecture

**Title:** Proposed Solution 2/4 - System Architecture (Required)

**Architecture Flow**
1. Data ingestion and validation
2. Knowledge graph storage (Neo4j INTERACTS_WITH edges)
3. Dataset construction (drug1, drug2, text, label)
4. Feature layer
   - Text: TF-IDF
   - Graph: topology + node-level graph descriptors
   - LLM: confidence estimates
5. Model training layer
   - Baseline ML, Graph ML, GNN, Hybrid
6. Evaluation and model comparison (Stage 7)
7. Deployment layer
   - Flask scoring app for pair-level inference

**Deployment Note**
- App endpoint scores user-entered drug pairs and returns model outputs.

**What to Visualize**
- Block diagram with 7 blocks above.
- Highlight train/test guardrails (group-aware split, threshold tuning).

---

## Slide 7 - Proposed Solution: Experiments Conducted

**Title:** Proposed Solution 3/4 - Experiment Design

**What We Trained (Experiment Tracks)**
- Traditional text models:
  - Logistic Regression (class-weight and SMOTE variants)
  - XGBoost (scale_pos_weight and SMOTE variants)
- Graph feature models:
  - Graph Logistic Regression
  - Graph XGBoost with extended topology features
- Graph neural model:
  - GraphSAGE edge classifier (message passing on train-only graph)
- Auxiliary model:
  - LLM-confidence predictor
- Fusion model:
  - Hybrid late-fusion combining text + graph + LLM channels

**What Changed Across Iterations**
- Switched to group-aware splits by canonical drug pair to reduce leakage.
- Expanded graph feature set (adamic-adar, resource allocation, salton, sorensen, support_count_log).
- Added threshold tuning on validation instead of fixed 0.5 threshold.
- Tested imbalance strategies (class weighting, scale_pos_weight, SMOTE) and retained stable variants.
- Added GNN pipeline with early stopping and reproducible seed.

**How We Evaluated Each Version**
- Same split protocol and metric set across runs for fair comparison.
- Metrics: Precision, Recall, F1, PR-AUC.
- Final selection based on deployment objective:
  - benchmark-optimized (best F1/PR-AUC)
  - safety-optimized (best recall)

---

## Slide 8 - Proposed Solution: Optimizations and Improvements

**Title:** Proposed Solution 4/4 - Key Optimizations

**Optimization Highlights**
- Threshold optimization objective:
  - 0.7 x macro-F1 + 0.3 x recall(class 0) in several pipelines
- Class imbalance handling:
  - class_weight and scale_pos_weight approaches
  - SMOTE tested and benchmarked
- Rich graph feature engineering:
  - adamic-adar, resource allocation, salton, sorensen, support_count_log
- GNN-specific training controls:
  - early stopping, regularization, fixed seed
- Hybrid fusion:
  - integrated text + graph + LLM confidence channels

**Outcome of Optimization Strategy**
- Weighted methods were generally stable under skew.
- Tested SMOTE settings were not consistently superior in final stage comparison.

---

## Slide 9 - Results and Analysis: Main Comparison

**Title:** Results 1/4 - Final Model Comparison (Stage 7)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 0.9488 | 0.9737 | 0.9997 |
| XGBoost | 1.0000 | 0.9331 | 0.9654 | 0.9886 |
| Graph Model | 0.9692 | 0.5294 | 0.6848 | 0.9626 |
| Graph GNN (GraphSAGE) | 0.9333 | 1.0000 | 0.9655 | 0.9547 |
| LLM Model | 0.9345 | 0.8425 | 0.8861 | 0.9305 |
| Hybrid (Late Fusion) | 0.9447 | 0.9409 | 0.9428 | 0.9533 |

**Primary Readout**
- Best F1 and PR-AUC: Logistic Regression.
- Best Recall: GraphSAGE GNN (perfect recall = 1.0000).

---

## Slide 10 - Results and Analysis: Safety-Critical Interpretation

**Title:** Results 2/4 - Why Recall Changes the Winner in Practice

**Key Interpretation**
- In pharmacovigilance screening, false negatives are often costlier than false positives.
- GNN achieved recall = 1.0000 while maintaining high precision (0.9333).
- Logistic Regression achieved top aggregate metrics but slightly lower recall.

**Actionable Interpretation**
- If objective is strict safety screening: favor GNN or recall-prioritized ensemble policy.
- If objective is balanced benchmark maximum: Logistic Regression is strongest.

---

## Slide 11 - Results and Analysis: Imbalance Effects

**Title:** Results 3/4 - Class Imbalance and Strategy Behavior

**Observed in evaluation summary**
- Raw train distribution example: 74 negative vs 1,018 positive.
- Balanced resampling example: 1,017 negative vs 1,018 positive.
- Noted effect: weighted methods remained stable; tested SMOTE setup degraded one hybrid-stage F1.

**Interpretation**
- Under heavy skew, calibration/thresholding and weighted losses can be more reliable than naive oversampling.
- Imbalance handling must be validated empirically per architecture.

---

## Slide 12 - Results and Analysis: Graph vs Non-Graph Takeaways

**Title:** Results 4/4 - Modality-Level Insights

**Findings**
- Graph-only classical features underperformed on recall (0.5294), limiting standalone utility.
- GNN message passing significantly improved graph-based recall behavior.
- LLM confidence channel is informative but weaker as a standalone predictor.
- Hybrid late fusion produced a balanced profile, but did not surpass best text baseline in this run.

**Interpretation**
- Structural information helps most when learned end-to-end (GNN), not only via handcrafted graph features.

---

## Slide 13 - Conclusion and Future Work

**Title:** Conclusion and Future Work 1/2

**Conclusions**
- Aura-Discovery successfully compared 6 model families for DDI prediction.
- Best benchmark model: Logistic Regression (F1 0.9737, PR-AUC 0.9997).
- Best safety-oriented model: GraphSAGE GNN (Recall 1.0000).
- Core lesson: model choice should align with deployment objective (benchmark optimization vs safety-first screening).

**Limitations**
- Severe class imbalance and sparse negatives.
- Pair-frequency concentration may limit generalization.
- LLM component had fallback-heavy usage in hybrid pipeline.

---

## Slide 14 - Conclusion and Future Work + Literature Positioning

**Title:** Conclusion and Future Work 2/2

**Relation to Relevant Literature (Positioning)**
- Consistent with prior evidence: graph neural methods often improve relational recall in biomedical link prediction tasks.
- This project confirms that high-recall graph learning can be competitive with strong text baselines.

**Future Improvements**
- Calibrated ensemble combining Logistic Regression confidence with GNN recall strength.
- Better negative sampling and hard-negative mining.
- Expanded multimodal context (mechanism pathways, dosage, temporal effects).
- External validation on independent DDI benchmark datasets.

---

## Slide 15 - References (Required)

**Title:** References

- This project processed 28 papers from the References folder.
- The slide below lists the most relevant sources in formal bibliography style; the remaining references are retained in the References folder for the full appendix.

**Selected formal references**
- Luo, H., Yin, W., Wang, J., Zhang, G., Liang, W., Luo, J., & Yan, C. (2024). Drug-drug interactions prediction based on deep learning and knowledge graph: A review. *iScience Review*. (`PIIS2589004224003699.pdf`)
- Li, X., et al. (2024). Deep learning for drug-drug interaction prediction: A comprehensive review. *Quantitative Biology*. (`Quant  Biol - 2024 - Li - Deep learning for drug‐drug interaction prediction  A comprehensive review.pdf`)
- Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10, 67. (`s41597-023-01960-3.pdf`)
- Al-Rabeah, M. H., & Lakizadeh, A. (2022). Prediction of drug-drug interaction events using graph neural networks based feature extraction. *Scientific Reports*, 12, 15590. (`s41598-022-19999-4.pdf`)
- Yu, C., Zhang, S., Wang, X., Shi, T., Jiang, C., Liang, S., & Ma, G. (2023). Drug-drug interaction extraction based on multimodal feature fusion by Transformer and BiGRU. *Frontiers in Pharmacology*. (`fddsv-04-1460672.pdf`)
- Machado, M., et al. (2023). Drug-drug interaction extraction-based system: An NLP approach. *Expert Systems*. (`Expert Systems - 2023 - Machado - Drug drug interaction extraction‐based system  An natural language processing approach.pdf`)
- Chen, Q., et al. (2025). Benchmarking large language models for biomedical natural language processing applications and recommendations. *Nature Communications*. (`s41467-025-56989-2.pdf`)
- Peng, B., Zhu, Y., Liu, Y., Bo, X., Shi, H., & Hong, C. (2025). Graph retrieval-augmented generation: A survey. *ACM Computing Surveys*. (`3777378.pdf`)
- Bian, J., et al. (2024). High-throughput biomedical relation extraction for LLM-based screening. (`High-throughput BioMed Relation.pdf`)
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*.

**Corpus note**
- Additional supporting papers from the References folder cover knowledge graphs, biomedical databases, GraphRAG, PubMed, DrugBank, ChEMBL, and class imbalance methods.

---

## Optional Appendix Slide - Backup

**Ablation/Comparison Snapshot**
- Show baseline vs graph vs hybrid metrics from artifacts/metrics.
- Include confusion-matrix snippets for top two deployment candidates.

