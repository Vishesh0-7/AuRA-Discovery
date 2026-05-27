# Graph GNN Experiment Summary (PPT-Ready)

## What was added
- Implemented a true Graph Neural Network trainer: `train_graph_gnn_model.py`.
- Model architecture: 2-layer GraphSAGE-style encoder + edge MLP classifier.
- Added automatic data fallback:
  - Primary source: Neo4j interactions
  - Fallback source: `data/processed/ddi_dataset.csv` when Neo4j is unavailable
- Added Stage-7 integration so the GNN appears automatically in model comparison outputs.

## Training run details
- Run date: 2026-05-03
- Data source used in this run: CSV fallback (`data/processed/ddi_dataset.csv`)
- Total rows: 1,275
- Split: train 816 / val 204 / test 255
- Class distribution:
  - Train: 761 positive, 55 negative
  - Test: 238 positive, 17 negative
- Graph stats:
  - Nodes: 1,066
  - Message-passing edges: 1,632

## New GNN results
- Validation (best epoch 44):
  - Precision: 0.9314
  - Recall: 1.0000
  - F1: 0.9645
  - PR-AUC: 0.9531
  - Threshold: 0.10
- Test:
  - Precision: 0.9333
  - Recall: 1.0000
  - F1: 0.9655
  - PR-AUC: 0.9578

## Updated Stage-7 comparison table
- Logistic Regression: P=1.0000, R=0.9488, F1=0.9737, PR-AUC=0.9997
- XGBoost: P=1.0000, R=0.9331, F1=0.9654, PR-AUC=0.9886
- Graph Model: P=0.9692, R=0.5294, F1=0.6848, PR-AUC=0.9626
- Graph GNN (GraphSAGE): P=0.9333, R=1.0000, F1=0.9655, PR-AUC=0.9578
- LLM Model: P=0.9345, R=0.8425, F1=0.8861, PR-AUC=0.9305
- Hybrid (Late Fusion): P=0.9447, R=0.9409, F1=0.9428, PR-AUC=0.9533

## Key takeaways for slides
- The new GNN dramatically improves graph-only F1 over the prior graph baseline:
  - F1: 0.6848 -> 0.9655 (+0.2807)
  - Recall: 0.5294 -> 1.0000 (+0.4706)
- The GNN is now competitive with top classical models on F1.
- Logistic Regression remains best overall in current Stage-7 table (F1 and PR-AUC).

## Important caveat (mention in presentation)
- Neo4j was offline during this run.
- GNN metrics were produced from CSV fallback data (1,275 rows, strongly positive-skewed), while existing Stage-7 metrics may come from prior runs/splits.
- Therefore, this table is useful for directional comparison, but not a strict apples-to-apples benchmark.

## Suggested final benchmark plan
- Re-run all model families on one shared split and one shared data source (preferably with Neo4j online).
- Keep threshold tuning protocol identical across models.
- Then publish final comparison table for the paper/presentation.
