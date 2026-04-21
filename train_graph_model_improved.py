"""Train improved graph-only DDI models and compare against baseline.

Enhancements rely on the upgraded graph feature pipeline, including:
- degree centrality
- PageRank
- shortest path length
- shared neighbors
- Jaccard similarity
- optional Node2Vec-derived features when dependencies are available

Outputs:
- artifacts/models/graph_improved_logreg.joblib
- artifacts/models/graph_improved_xgb.joblib
- artifacts/models/graph_improved_champion.joblib
- artifacts/models/graph_improved_feature_bundle.joblib
- artifacts/metrics/graph_metrics_improved.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import load_graph_training_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _build_logreg(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_xgb(y_train: np.ndarray, random_state: int) -> XGBClassifier:
    class_0 = max(1, int((y_train == 0).sum()))
    class_1 = max(1, int((y_train == 1).sum()))
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=class_0 / class_1,
        random_state=random_state,
        n_jobs=4,
    )


def _load_graph_baseline_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    champion_name = data.get("champion", {}).get("model_name")
    if not champion_name:
        return {}

    model_test = data.get("models", {}).get(champion_name, {}).get("test", {})
    if not model_test:
        return {}

    return {
        "model": champion_name,
        "precision": float(model_test.get("precision_class_1", 0.0)),
        "recall": float(model_test.get("recall_class_1", 0.0)),
        "f1": float(model_test.get("f1_class_1", 0.0)),
        "pr_auc": float(model_test.get("pr_auc", 0.0)),
    }


def train_graph_model_improved(
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Dict[str, Any]:
    with ResearchGraph() as graph:
        bundle = load_graph_training_data(
            graph=graph,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

    x_train = bundle["x_train"]
    y_train = bundle["y_train"]
    x_val = bundle["x_val"]
    y_val = bundle["y_val"]
    x_test = bundle["x_test"]
    y_test = bundle["y_test"]

    logreg = _build_logreg(random_state=random_state)
    logreg.fit(x_train, y_train)
    logreg_val_prob = logreg.predict_proba(x_val)[:, 1]
    logreg_test_prob = logreg.predict_proba(x_test)[:, 1]

    xgb = _build_xgb(y_train=y_train, random_state=random_state)
    xgb.fit(x_train, y_train)
    xgb_val_prob = xgb.predict_proba(x_val)[:, 1]
    xgb_test_prob = xgb.predict_proba(x_test)[:, 1]

    models = {
        "graph_improved_logreg": {
            "validation": _evaluate(y_val, logreg_val_prob),
            "test": _evaluate(y_test, logreg_test_prob),
        },
        "graph_improved_xgb": {
            "validation": _evaluate(y_val, xgb_val_prob),
            "test": _evaluate(y_test, xgb_test_prob),
        },
    }

    champion_name = max(models, key=lambda key: (models[key]["validation"]["f1"], models[key]["validation"]["pr_auc"]))
    champion_model = logreg if champion_name == "graph_improved_logreg" else xgb
    champion_test = models[champion_name]["test"]

    baseline = _load_graph_baseline_metrics(Path("artifacts/metrics/graph_metrics.json"))
    improvement = {}
    if baseline:
        improvement = {
            "baseline_model": baseline["model"],
            "improved_model": champion_name,
            "delta_precision": float(champion_test["precision"] - baseline["precision"]),
            "delta_recall": float(champion_test["recall"] - baseline["recall"]),
            "delta_f1": float(champion_test["f1"] - baseline["f1"]),
            "delta_pr_auc": float(champion_test["pr_auc"] - baseline["pr_auc"]),
        }

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(logreg, models_dir / "graph_improved_logreg.joblib")
    joblib.dump(xgb, models_dir / "graph_improved_xgb.joblib")
    joblib.dump(
        {
            "model_name": champion_name,
            "model": champion_model,
        },
        models_dir / "graph_improved_champion.joblib",
    )
    joblib.dump(
        {
            "feature_names": bundle["feature_names"],
            "neighbor_map": bundle["neighbor_map"],
        },
        models_dir / "graph_improved_feature_bundle.joblib",
    )

    report = {
        "config": {
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
        },
        "dataset": {
            "total_rows": len(bundle["rows"]),
            "train_rows": len(bundle["train_rows"]),
            "val_rows": len(bundle["val_rows"]),
            "test_rows": len(bundle["test_rows"]),
            "train_class_0": int((y_train == 0).sum()),
            "train_class_1": int((y_train == 1).sum()),
            "test_class_0": int((y_test == 0).sum()),
            "test_class_1": int((y_test == 1).sum()),
            "feature_count": int(x_train.shape[1]),
            "feature_names": bundle["feature_names"],
        },
        "models": models,
        "champion": {
            "model_name": champion_name,
            "validation": models[champion_name]["validation"],
            "test": champion_test,
        },
        "baseline_comparison": {
            "baseline": baseline,
            "improvement": improvement,
        },
    }

    metrics_path = metrics_dir / "graph_metrics_improved.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Saved improved graph model metrics to %s", metrics_path)
    logger.info(
        "Champion=%s test precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        champion_name,
        champion_test["precision"],
        champion_test["recall"],
        champion_test["f1"],
        champion_test["pr_auc"],
    )
    if improvement:
        logger.info(
            "Improvement vs baseline: dF1=%.4f dPR-AUC=%.4f",
            improvement["delta_f1"],
            improvement["delta_pr_auc"],
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved graph-only DDI model")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    train_graph_model_improved(
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()