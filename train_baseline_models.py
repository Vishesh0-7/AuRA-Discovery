"""Train improved baseline DDI models with imbalance-aware evaluation.

Improvements over naive baseline:
- Group-aware split by canonical drug pair to reduce leakage.
- Validation-based threshold tuning (not fixed 0.5).
- Weighted and SMOTE training variants.
- Rich imbalance metrics for both classes.

Models trained:
- Logistic Regression (class-weighted)
- Logistic Regression (SMOTE)
- XGBoost (sample-weighted)
- XGBoost (SMOTE)

Outputs:
- artifacts/models/feature_pipeline.joblib
- artifacts/models/*.joblib (one per model)
- artifacts/models/champion_model.joblib
- artifacts/metrics/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
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
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from src.features.pipeline import DDIRow, build_feature_pipeline, read_ddi_dataset, transform_rows


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _canonical_pair(drug1: str, drug2: str) -> str:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    a, b = sorted((left, right))
    return f"{a}||{b}"


def _rows_by_indices(rows: Sequence[DDIRow], indices: np.ndarray) -> List[DDIRow]:
    return [rows[int(i)] for i in indices]


def _group_stratified_split(
    y: np.ndarray,
    groups: Sequence[str],
    random_state: int,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    x_dummy = np.zeros(len(y), dtype=np.int8)
    train_idx, test_idx = next(splitter.split(x_dummy, y, groups))
    return train_idx, test_idx


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
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


def _threshold_objective(metrics: Dict[str, Any]) -> float:
    # Give macro-F1 primary weight while explicitly rewarding minority recall.
    return (0.7 * metrics["macro_f1"]) + (0.3 * metrics["recall_class_0"])


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    best_metrics: Dict[str, Any] | None = None
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = _compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))
        score = _threshold_objective(metrics)

        if best_metrics is None:
            best_metrics = metrics
            best_score = score
            continue

        # Tie-break by stronger minority recall when objective ties.
        if score > best_score or (
            abs(score - best_score) < 1e-12
            and metrics["recall_class_0"] > best_metrics["recall_class_0"]
        ):
            best_metrics = metrics
            best_score = score

    if best_metrics is None:
        raise RuntimeError("Threshold search failed to produce metrics")

    return best_metrics


def _predict_proba(model: Any, x_data: Any) -> np.ndarray:
    return model.predict_proba(x_data)[:, 1]


def _build_logreg(config: Dict[str, Any], random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=config.get("C", 1.0),
        max_iter=3000,
        solver="liblinear",
        class_weight=config.get("class_weight"),
        random_state=random_state,
    )


def _build_xgb(config: Dict[str, Any], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=config.get("n_estimators", 500),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 6),
        min_child_weight=config.get("min_child_weight", 2),
        subsample=config.get("subsample", 0.9),
        colsample_bytree=config.get("colsample_bytree", 0.9),
        reg_lambda=config.get("reg_lambda", 1.0),
        random_state=random_state,
        n_jobs=4,
    )


def _inverse_frequency_weights(y: np.ndarray) -> np.ndarray:
    class_0 = int((y == 0).sum())
    class_1 = int((y == 1).sum())
    total = len(y)
    w0 = total / (2 * class_0)
    w1 = total / (2 * class_1)
    return np.where(y == 0, w0, w1).astype(np.float32)


def _smote_resample(
    x_train_dense: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    minority_count = int((y_train == 0).sum())

    if minority_count <= 1:
        raise ValueError("SMOTE requires at least two minority samples in y_train")

    k_neighbors = min(5, minority_count - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    x_resampled, y_resampled = smote.fit_resample(x_train_dense, y_train)

    distribution = {
        "before_class_0": int((y_train == 0).sum()),
        "before_class_1": int((y_train == 1).sum()),
        "after_class_0": int((y_resampled == 0).sum()),
        "after_class_1": int((y_resampled == 1).sum()),
    }
    return x_resampled, y_resampled, distribution


def train_baselines(
    data_path: str = "data/processed/ddi_dataset.csv",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    max_tfidf_features: int = 20000,
) -> Dict[str, Any]:
    rows: List[DDIRow] = read_ddi_dataset(data_path)
    if len(rows) < 10:
        raise ValueError(f"Not enough data rows for training: {len(rows)}")

    labels = np.array([row.label for row in rows], dtype=np.int64)
    groups = np.array([_canonical_pair(row.drug1, row.drug2) for row in rows], dtype=object)

    # Hold out test set with group-aware stratification.
    train_val_idx, test_idx = _group_stratified_split(
        y=labels,
        groups=groups,
        random_state=random_state,
        n_splits=max(2, int(round(1.0 / test_size))),
    )

    train_val_rows = _rows_by_indices(rows, train_val_idx)
    y_train_val = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]

    # Validation split for threshold tuning and model comparison.
    train_idx_rel, val_idx_rel = _group_stratified_split(
        y=y_train_val,
        groups=groups_train_val,
        random_state=random_state + 1,
        n_splits=max(2, int(round(1.0 / val_size))),
    )

    train_rows = _rows_by_indices(train_val_rows, train_idx_rel)
    val_rows = _rows_by_indices(train_val_rows, val_idx_rel)
    test_rows = _rows_by_indices(rows, test_idx)

    x_train_sparse, y_train, feature_pipeline = build_feature_pipeline(
        rows=train_rows,
        max_tfidf_features=max_tfidf_features,
    )
    x_val_sparse, y_val = transform_rows(val_rows, feature_pipeline)
    x_test_sparse, y_test = transform_rows(test_rows, feature_pipeline)

    x_train_dense = x_train_sparse.toarray().astype(np.float32)
    x_val_dense = x_val_sparse.toarray().astype(np.float32)
    x_test_dense = x_test_sparse.toarray().astype(np.float32)

    x_train_smote, y_train_smote, smote_dist = _smote_resample(
        x_train_dense=x_train_dense,
        y_train=y_train,
        random_state=random_state,
    )

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    experiments: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Any] = {}

    experiment_configs: Dict[str, Dict[str, Any]] = {
        "logreg_weighted": {
            "strategy": "class_weight",
            "use_dense": False,
            "builder": _build_logreg,
            "params": [
                {"C": 0.5, "class_weight": "balanced"},
                {"C": 1.0, "class_weight": "balanced"},
                {"C": 2.0, "class_weight": "balanced"},
            ],
        },
        "logreg_smote": {
            "strategy": "smote",
            "use_dense": True,
            "builder": _build_logreg,
            "params": [
                {"C": 0.5, "class_weight": None},
                {"C": 1.0, "class_weight": None},
                {"C": 2.0, "class_weight": None},
            ],
        },
        "xgb_weighted": {
            "strategy": "sample_weight",
            "use_dense": False,
            "builder": _build_xgb,
            "params": [
                {
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "min_child_weight": 2,
                },
                {
                    "n_estimators": 600,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "min_child_weight": 3,
                },
            ],
        },
        "xgb_smote": {
            "strategy": "smote",
            "use_dense": True,
            "builder": _build_xgb,
            "params": [
                {
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "min_child_weight": 2,
                },
                {
                    "n_estimators": 600,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "min_child_weight": 3,
                },
            ],
        },
    }

    def cv_objective_for_params(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        train_labels_local = np.array([row.label for row in train_rows], dtype=np.int64)
        train_groups_local = np.array([_canonical_pair(row.drug1, row.drug2) for row in train_rows], dtype=object)

        cv_splitter = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=random_state,
        )
        x_dummy = np.zeros(len(train_rows), dtype=np.int8)

        fold_objectives: List[float] = []
        fold_thresholds: List[float] = []

        for fold_id, (fold_train_idx, fold_val_idx) in enumerate(
            cv_splitter.split(x_dummy, train_labels_local, train_groups_local),
            start=1,
        ):
            fold_train_rows = _rows_by_indices(train_rows, fold_train_idx)
            fold_val_rows = _rows_by_indices(train_rows, fold_val_idx)

            fold_x_train_sparse, fold_y_train, fold_pipeline = build_feature_pipeline(
                rows=fold_train_rows,
                max_tfidf_features=max_tfidf_features,
            )
            fold_x_val_sparse, fold_y_val = transform_rows(fold_val_rows, fold_pipeline)

            if config["use_dense"]:
                fold_x_train = fold_x_train_sparse.toarray().astype(np.float32)
                fold_x_val = fold_x_val_sparse.toarray().astype(np.float32)
            else:
                fold_x_train = fold_x_train_sparse
                fold_x_val = fold_x_val_sparse

            model = config["builder"](params, random_state + fold_id)

            if config["strategy"] == "sample_weight":
                fold_weights = _inverse_frequency_weights(fold_y_train)
                model.fit(fold_x_train, fold_y_train, sample_weight=fold_weights)
            elif config["strategy"] == "smote":
                fold_x_train_dense = fold_x_train_sparse.toarray().astype(np.float32)
                fold_x_smote, fold_y_smote, _ = _smote_resample(
                    x_train_dense=fold_x_train_dense,
                    y_train=fold_y_train,
                    random_state=random_state + fold_id,
                )
                model.fit(fold_x_smote, fold_y_smote)
            else:
                model.fit(fold_x_train, fold_y_train)

            fold_y_val_prob = _predict_proba(model, fold_x_val)
            fold_best = _find_best_threshold(fold_y_val, fold_y_val_prob)
            fold_objectives.append(_threshold_objective(fold_best))
            fold_thresholds.append(float(fold_best["threshold"]))

        return {
            "mean_objective": float(np.mean(fold_objectives)),
            "std_objective": float(np.std(fold_objectives)),
            "mean_threshold": float(np.mean(fold_thresholds)),
        }

    def run_experiment(name: str, config: Dict[str, Any]) -> None:
        x_train_fit = x_train_dense if config["use_dense"] else x_train_sparse
        x_val_eval = x_val_dense if config["use_dense"] else x_val_sparse
        x_test_eval = x_test_dense if config["use_dense"] else x_test_sparse

        best_variant: Dict[str, Any] | None = None
        best_model: Any | None = None
        best_objective = -1.0

        for params in config["params"]:
            cv_summary = cv_objective_for_params(config, params)

            model = config["builder"](params, random_state)

            if config["strategy"] == "sample_weight":
                sample_weights = _inverse_frequency_weights(y_train)
                model.fit(x_train_fit, y_train, sample_weight=sample_weights)
            elif config["strategy"] == "smote":
                model.fit(x_train_smote, y_train_smote)
            else:
                model.fit(x_train_fit, y_train)

            y_val_prob = _predict_proba(model, x_val_eval)
            best_val = _find_best_threshold(y_val, y_val_prob)
            objective = cv_summary["mean_objective"]

            if best_variant is None or objective > best_objective:
                best_variant = {
                    "params": params,
                    "validation": best_val,
                    "cv": cv_summary,
                }
                best_model = model
                best_objective = objective

        if best_variant is None or best_model is None:
            raise RuntimeError(f"No model variant trained for experiment {name}")

        y_test_prob = _predict_proba(best_model, x_test_eval)
        test_metrics = _compute_binary_metrics(
            y_true=y_test,
            y_prob=y_test_prob,
            threshold=best_variant["validation"]["threshold"],
        )

        experiments[name] = {
            "training_strategy": config["strategy"],
            "best_params": best_variant["params"],
            "cv": best_variant["cv"],
            "validation": best_variant["validation"],
            "test": test_metrics,
            "selection_objective": float(best_objective),
        }
        trained_models[name] = best_model

    for experiment_name, experiment_cfg in experiment_configs.items():
        run_experiment(experiment_name, experiment_cfg)

    champion_name = max(
        experiments,
        key=lambda model_name: (
            experiments[model_name]["validation"]["macro_f1"],
            experiments[model_name]["validation"]["recall_class_0"],
        ),
    )

    for model_name, model in trained_models.items():
        joblib.dump(model, models_dir / f"{model_name}.joblib")

    champion_artifact = {
        "model_name": champion_name,
        "threshold": experiments[champion_name]["validation"]["threshold"],
        "model": trained_models[champion_name],
    }
    joblib.dump(champion_artifact, models_dir / "champion_model.joblib")
    joblib.dump(feature_pipeline, models_dir / "feature_pipeline.joblib")

    results = {
        "config": {
            "data_path": data_path,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "max_tfidf_features": max_tfidf_features,
            "threshold_selection_metric": "0.7*macro_f1 + 0.3*recall_class_0",
        },
        "dataset": {
            "total_rows": len(rows),
            "train_val_rows": len(train_val_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_class_0": int((y_train == 0).sum()),
            "train_class_1": int((y_train == 1).sum()),
            "val_class_0": int((y_val == 0).sum()),
            "val_class_1": int((y_val == 1).sum()),
            "test_class_0": int((y_test == 0).sum()),
            "test_class_1": int((y_test == 1).sum()),
            "feature_count": int(x_train_sparse.shape[1]),
            "unique_groups_total": int(len(set(groups.tolist()))),
            "smote_distribution": smote_dist,
        },
        "models": experiments,
        "champion": {
            "model_name": champion_name,
            "threshold": experiments[champion_name]["validation"]["threshold"],
            "validation_macro_f1": experiments[champion_name]["validation"]["macro_f1"],
            "test_macro_f1": experiments[champion_name]["test"]["macro_f1"],
        },
    }

    metrics_path = metrics_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    logger.info("Saved feature pipeline to %s", models_dir / "feature_pipeline.joblib")
    logger.info("Saved champion model to %s", models_dir / "champion_model.joblib")
    logger.info("Saved metrics to %s", metrics_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved DDI baselines with imbalance handling")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-tfidf-features", type=int, default=20000)
    args = parser.parse_args()

    metrics = train_baselines(
        data_path=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        max_tfidf_features=args.max_tfidf_features,
    )

    champion = metrics["champion"]
    logger.info(
        "Champion: %s | val macro-F1=%.4f | test macro-F1=%.4f | threshold=%.2f",
        champion["model_name"],
        champion["validation_macro_f1"],
        champion["test_macro_f1"],
        champion["threshold"],
    )


if __name__ == "__main__":
    main()
