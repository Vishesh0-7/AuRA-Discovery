"""Train a graph-based DDI model from Neo4j topology features.

Features used:
- degree
- common neighbors
- Jaccard similarity
- support counts

Outputs:
- artifacts/models/graph_feature_bundle.joblib
- artifacts/models/graph_logreg.joblib
- artifacts/models/graph_xgb.joblib
- artifacts/models/graph_champion.joblib
- artifacts/metrics/graph_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import load_graph_training_data


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _threshold_objective(metrics: Dict[str, Any]) -> float:
    return (0.7 * metrics["macro_f1"]) + (0.3 * metrics["recall_class_0"])


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_class_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_class_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_class_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_class_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_class_0": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    best_metrics: Dict[str, Any] | None = None
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = _compute_metrics(y_true, y_prob, float(threshold))
        score = _threshold_objective(metrics)
        if best_metrics is None or score > best_score:
            best_metrics = metrics
            best_score = score
        elif abs(score - best_score) < 1e-12 and metrics["recall_class_0"] > best_metrics["recall_class_0"]:
            best_metrics = metrics
            best_score = score

    if best_metrics is None:
        raise RuntimeError("Threshold search failed")
    return best_metrics


def _fit_logistic_regression(x_train: np.ndarray, y_train: np.ndarray, random_state: int) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def _fit_logistic_regression_with_c(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    c_value: float,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def _fit_xgboost(x_train: np.ndarray, y_train: np.ndarray, random_state: int) -> XGBClassifier:
    class_0 = max(1, int((y_train == 0).sum()))
    class_1 = max(1, int((y_train == 1).sum()))
    scale_pos_weight = class_0 / class_1

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model


def _fit_xgboost_with_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    params: Dict[str, Any],
) -> XGBClassifier:
    class_0 = max(1, int((y_train == 0).sum()))
    class_1 = max(1, int((y_train == 1).sum()))
    scale_pos_weight = class_0 / class_1

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model


def train_graph_model(
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

    logreg_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    best_logreg: Pipeline | None = None
    best_logreg_params: Dict[str, Any] | None = None
    best_logreg_val: Dict[str, Any] | None = None
    best_logreg_objective = -1.0

    for c_value in logreg_candidates:
        candidate = _fit_logistic_regression_with_c(x_train, y_train, random_state, c_value)
        candidate_val_prob = candidate.predict_proba(x_val)[:, 1]
        candidate_best_val = _find_best_threshold(y_val, candidate_val_prob)
        objective = _threshold_objective(candidate_best_val)
        if objective > best_logreg_objective:
            best_logreg_objective = objective
            best_logreg = candidate
            best_logreg_params = {"C": c_value}
            best_logreg_val = candidate_best_val

    xgb_candidates = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 2},
        {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 2},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 3},
    ]
    best_xgb: XGBClassifier | None = None
    best_xgb_params: Dict[str, Any] | None = None
    best_xgb_val: Dict[str, Any] | None = None
    best_xgb_objective = -1.0

    for params in xgb_candidates:
        candidate = _fit_xgboost_with_params(x_train, y_train, random_state, params)
        candidate_val_prob = candidate.predict_proba(x_val)[:, 1]
        candidate_best_val = _find_best_threshold(y_val, candidate_val_prob)
        objective = _threshold_objective(candidate_best_val)
        if objective > best_xgb_objective:
            best_xgb_objective = objective
            best_xgb = candidate
            best_xgb_params = params
            best_xgb_val = candidate_best_val

    if best_logreg is None or best_logreg_val is None or best_logreg_params is None:
        raise RuntimeError("Logistic regression search failed")
    if best_xgb is None or best_xgb_val is None or best_xgb_params is None:
        raise RuntimeError("XGBoost search failed")

    logreg_test_prob = best_logreg.predict_proba(x_test)[:, 1]
    xgb_test_prob = best_xgb.predict_proba(x_test)[:, 1]

    experiments = {
        "graph_logreg": {
            "best_params": best_logreg_params,
            "selection_objective": float(best_logreg_objective),
            "validation": best_logreg_val,
            "test": _compute_metrics(y_test, logreg_test_prob, best_logreg_val["threshold"]),
        },
        "graph_xgb": {
            "best_params": best_xgb_params,
            "selection_objective": float(best_xgb_objective),
            "validation": best_xgb_val,
            "test": _compute_metrics(y_test, xgb_test_prob, best_xgb_val["threshold"]),
        },
    }

    champion_name = max(
        experiments,
        key=lambda name: (
            experiments[name]["validation"]["macro_f1"],
            experiments[name]["validation"]["recall_class_0"],
        ),
    )
    champion_model = best_logreg if champion_name == "graph_logreg" else best_xgb
    champion_threshold = experiments[champion_name]["validation"]["threshold"]

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "neighbor_map": bundle["neighbor_map"],
            "feature_names": bundle["feature_names"],
        },
        models_dir / "graph_feature_bundle.joblib",
    )
    joblib.dump(best_logreg, models_dir / "graph_logreg.joblib")
    joblib.dump(best_xgb, models_dir / "graph_xgb.joblib")
    joblib.dump(
        {
            "model_name": champion_name,
            "threshold": champion_threshold,
            "model": champion_model,
        },
        models_dir / "graph_champion.joblib",
    )

    metrics = {
        "config": {
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "threshold_selection_metric": "0.7*macro_f1 + 0.3*recall_class_0",
        },
        "dataset": {
            "total_rows": len(bundle["rows"]),
            "train_rows": len(bundle["train_rows"]),
            "val_rows": len(bundle["val_rows"]),
            "test_rows": len(bundle["test_rows"]),
            "train_class_0": int((y_train == 0).sum()),
            "train_class_1": int((y_train == 1).sum()),
            "val_class_0": int((y_val == 0).sum()),
            "val_class_1": int((y_val == 1).sum()),
            "test_class_0": int((y_test == 0).sum()),
            "test_class_1": int((y_test == 1).sum()),
            "feature_names": bundle["feature_names"],
        },
        "models": experiments,
        "champion": {
            "model_name": champion_name,
            "threshold": champion_threshold,
            "validation_macro_f1": experiments[champion_name]["validation"]["macro_f1"],
            "test_macro_f1": experiments[champion_name]["test"]["macro_f1"],
        },
    }

    metrics_path = metrics_dir / "graph_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Saved graph feature bundle to %s", models_dir / "graph_feature_bundle.joblib")
    logger.info("Saved graph champion to %s", models_dir / "graph_champion.joblib")
    logger.info("Saved graph metrics to %s", metrics_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph-based DDI model from Neo4j")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    metrics = train_graph_model(
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    champion = metrics["champion"]
    logger.info(
        "Graph champion: %s | val macro-F1=%.4f | test macro-F1=%.4f | threshold=%.2f",
        champion["model_name"],
        champion["validation_macro_f1"],
        champion["test_macro_f1"],
        champion["threshold"],
    )


if __name__ == "__main__":
    main()
