"""Stage 3: Train traditional ML models with imbalance correction.

Models:
- Logistic Regression (uses class_weight='balanced')
- XGBoost (uses scale_pos_weight)

Feature source:
- TF-IDF text features
- drug-pair encoding features
- normalized numeric pair features

Outputs:
- artifacts/models/traditional_logreg.joblib
- artifacts/models/traditional_xgb.joblib
- artifacts/models/traditional_feature_pipeline.joblib
- artifacts/metrics/traditional_ml_metrics.json
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
from xgboost import XGBClassifier

from src.features.pipeline import MLReadySplit, load_ml_ready_split


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _train_logreg(split: MLReadySplit, random_state: int) -> tuple[LogisticRegression, Dict[str, Any]]:
    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=random_state,
    )
    model.fit(split.x_train, split.y_train)

    y_prob = model.predict_proba(split.x_test)[:, 1]
    metrics = _evaluate_binary(split.y_test, y_prob)
    metrics["model"] = "logistic_regression"
    return model, metrics


def _train_xgb(split: MLReadySplit, random_state: int) -> tuple[XGBClassifier, Dict[str, Any]]:
    negatives = max(1, int((split.y_train == 0).sum()))
    positives = max(1, int((split.y_train == 1).sum()))
    scale_pos_weight = negatives / positives

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=4,
    )
    model.fit(split.x_train, split.y_train)

    y_prob = model.predict_proba(split.x_test)[:, 1]
    metrics = _evaluate_binary(split.y_test, y_prob)
    metrics["model"] = "xgboost"
    metrics["scale_pos_weight"] = float(scale_pos_weight)
    return model, metrics


def train_traditional_models(
    data_path: str = "data/processed/ddi_dataset_balanced.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    max_tfidf_features: int = 5000,
) -> Dict[str, Any]:
    split = load_ml_ready_split(
        csv_path=data_path,
        test_size=test_size,
        random_state=random_state,
        max_tfidf_features=max_tfidf_features,
    )

    logreg_model, logreg_metrics = _train_logreg(split, random_state=random_state)
    xgb_model, xgb_metrics = _train_xgb(split, random_state=random_state)

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(logreg_model, models_dir / "traditional_logreg.joblib")
    joblib.dump(xgb_model, models_dir / "traditional_xgb.joblib")
    joblib.dump(split.pipeline, models_dir / "traditional_feature_pipeline.joblib")

    metrics = {
        "config": {
            "data_path": data_path,
            "test_size": test_size,
            "random_state": random_state,
            "max_tfidf_features": max_tfidf_features,
        },
        "dataset": {
            "train_rows": int(split.x_train.shape[0]),
            "test_rows": int(split.x_test.shape[0]),
            "feature_count": int(split.x_train.shape[1]),
            "train_positive": int((split.y_train == 1).sum()),
            "train_negative": int((split.y_train == 0).sum()),
            "test_positive": int((split.y_test == 1).sum()),
            "test_negative": int((split.y_test == 0).sum()),
        },
        "models": {
            "logistic_regression": logreg_metrics,
            "xgboost": xgb_metrics,
        },
    }

    metrics_path = metrics_dir / "traditional_ml_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Saved model artifacts to %s", models_dir)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "LogReg: precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        logreg_metrics["precision"],
        logreg_metrics["recall"],
        logreg_metrics["f1"],
        logreg_metrics["pr_auc"],
    )
    logger.info(
        "XGBoost: precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        xgb_metrics["precision"],
        xgb_metrics["recall"],
        xgb_metrics["f1"],
        xgb_metrics["pr_auc"],
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 3 traditional ML models")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset_balanced.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-tfidf-features", type=int, default=5000)
    args = parser.parse_args()

    train_traditional_models(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        max_tfidf_features=args.max_tfidf_features,
    )


if __name__ == "__main__":
    main()