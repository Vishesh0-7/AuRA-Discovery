| Model | Precision | Recall | F1 | PR-AUC | Notes |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 1.0000 | 0.9488 | 0.9737 | 0.9997 | class_weight=balanced |
| XGBoost | 1.0000 | 0.9331 | 0.9654 | 0.9886 | scale_pos_weight |
| Graph Model | 0.9692 | 0.5294 | 0.6848 | 0.9626 | champion=graph_xgb |
| LLM Model | 0.9345 | 0.8425 | 0.8861 | 0.9305 | LLM-confidence model |
| Hybrid (Late Fusion) | 0.9447 | 0.9409 | 0.9428 | 0.9533 | TF-IDF + Graph + LLM |
