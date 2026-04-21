"""Train Hybrid Model D: TF-IDF + graph features + LLM output.

This script builds a hybrid feature matrix composed of:
1. Text features from TF-IDF + drug-pair text pipeline.
2. Graph topology features derived from Neo4j interactions.
3. LLM prediction features from Stage 5 predictor.

It trains an XGBoost model and writes a comparison report against the
existing traditional ML and graph-only model metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.agents.predictor import predict_llm_safe
from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import fetch_graph_rows
from src.features.pipeline import DDIRow, build_feature_pipeline, read_ddi_dataset, transform_rows


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    return tuple(sorted((left, right)))


def _rows_by_indices(rows: Sequence[DDIRow], indices: np.ndarray) -> List[DDIRow]:
    return [rows[int(i)] for i in indices]


def _group_stratified_split(
    y: np.ndarray,
    groups: Sequence[str],
    random_state: int,
    n_splits: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    x_dummy = np.zeros(len(y), dtype=np.int8)
    train_idx, holdout_idx = next(splitter.split(x_dummy, y, groups))
    return train_idx, holdout_idx


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
    best_objective = -1.0

    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = _compute_metrics(y_true, y_prob, float(threshold))
        objective = _threshold_objective(metrics)
        if best_metrics is None or objective > best_objective:
            best_metrics = metrics
            best_objective = objective
        elif abs(objective - best_objective) < 1e-12 and metrics["recall_class_0"] > best_metrics["recall_class_0"]:
            best_metrics = metrics
            best_objective = objective

    if best_metrics is None:
        raise RuntimeError("Threshold search failed")
    return best_metrics


def _build_graph_context() -> Dict[str, Any]:
    with ResearchGraph() as graph:
        graph_rows = fetch_graph_rows(graph)

    neighbor_map: Dict[str, set[str]] = {}
    support_count: Dict[Tuple[str, str], float] = {}

    for row in graph_rows:
        pair = (row.drug1, row.drug2)
        support_count[pair] = float(row.support_count)

        neighbor_map.setdefault(row.drug1, set()).add(row.drug2)
        neighbor_map.setdefault(row.drug2, set()).add(row.drug1)

    return {
        "neighbor_map": neighbor_map,
        "support_count": support_count,
    }


def _graph_features_for_pair(
    drug1: str,
    drug2: str,
    neighbor_map: Dict[str, set[str]],
    support_count: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    pair = _canonical_pair(drug1, drug2)
    d1, d2 = pair
    n1 = set(neighbor_map.get(d1, set())) - {d2}
    n2 = set(neighbor_map.get(d2, set())) - {d1}

    degree_1 = float(len(n1))
    degree_2 = float(len(n2))
    common = n1 & n2
    common_count = float(len(common))
    union_count = float(len(n1 | n2))
    jaccard = common_count / union_count if union_count else 0.0
    support = float(support_count.get(pair, 0.0))

    return {
        "graph_degree_1": degree_1,
        "graph_degree_2": degree_2,
        "graph_degree_sum": degree_1 + degree_2,
        "graph_common_neighbors": common_count,
        "graph_jaccard": jaccard,
        "graph_support_count": support,
    }


def _hash_row_key(drug1: str, drug2: str, text: str) -> str:
    key = "||".join([drug1.strip().lower(), drug2.strip().lower(), text.strip().lower()])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _load_llm_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_llm_cache(cache_path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _llm_features_for_rows(
    rows: Sequence[DDIRow],
    cache_path: Path,
    max_new_calls: int,
    model_name: str,
    priority_probabilities: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cache = _load_llm_cache(cache_path)
    new_calls = 0
    cache_hits = 0
    fallback_count = 0
    features: List[List[float]] = []

    query_order = list(range(len(rows)))
    if priority_probabilities is not None and len(priority_probabilities) == len(rows):
        query_order = sorted(
            query_order,
            key=lambda idx: abs(float(priority_probabilities[idx]) - 0.5),
        )

    selected_indices = set(query_order[:max_new_calls])

    for index, row in enumerate(rows):
        row_key = _hash_row_key(row.drug1, row.drug2, row.text)

        if row_key in cache:
            pred = cache[row_key]
            cache_hits += 1
        elif index in selected_indices:
            llm_result = predict_llm_safe(
                drug1=row.drug1,
                drug2=row.drug2,
                text=row.text,
                model_name=model_name,
                temperature=0.0,
            )
            pred = {
                "label": llm_result.label,
                "confidence": float(llm_result.confidence),
            }
            cache[row_key] = pred
            new_calls += 1
        else:
            pred = {
                "label": "No DDI",
                "confidence": 0.5,
            }
            fallback_count += 1

        label_num = 1.0 if pred.get("label") == "DDI" else 0.0
        confidence = float(pred.get("confidence", 0.5))
        signed_score = confidence if label_num > 0.5 else -confidence
        features.append([label_num, confidence, signed_score])

    _save_llm_cache(cache_path, cache)

    stats = {
        "cache_hits": cache_hits,
        "new_calls": new_calls,
        "fallback_count": fallback_count,
        "cache_size": len(cache),
    }

    return np.asarray(features, dtype=np.float32), stats


def _build_hybrid_matrix(
    rows: Sequence[DDIRow],
    x_text: csr_matrix,
    graph_context: Dict[str, Any],
    llm_cache_path: Path,
    max_llm_calls: int,
    llm_model_name: str,
    priority_probabilities: np.ndarray | None = None,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, Any]]:
    graph_feature_rows: List[List[float]] = []
    for row in rows:
        gf = _graph_features_for_pair(
            row.drug1,
            row.drug2,
            graph_context["neighbor_map"],
            graph_context["support_count"],
        )
        graph_feature_rows.append([
            gf["graph_degree_1"],
            gf["graph_degree_2"],
            gf["graph_degree_sum"],
            gf["graph_common_neighbors"],
            gf["graph_jaccard"],
            gf["graph_support_count"],
        ])

    x_graph = np.asarray(graph_feature_rows, dtype=np.float32)
    x_llm, llm_stats = _llm_features_for_rows(
        rows=rows,
        cache_path=llm_cache_path,
        max_new_calls=max_llm_calls,
        model_name=llm_model_name,
        priority_probabilities=priority_probabilities,
    )

    x_dense = np.concatenate([x_graph, x_llm], axis=1)
    x_dense_sparse = csr_matrix(x_dense)
    x_hybrid = hstack([x_text, x_dense_sparse], format="csr")

    return x_hybrid, x_dense, llm_stats


def _fit_xgb_with_params(
    x_train: csr_matrix,
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


def _load_existing_model_scores() -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}

    baseline_path = Path("artifacts/metrics/baseline_metrics.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        champion = baseline.get("champion", {})
        summaries["traditional_champion"] = {
            "model_name": champion.get("model_name"),
            "test_macro_f1": champion.get("test_macro_f1"),
        }

    graph_path = Path("artifacts/metrics/graph_metrics.json")
    if graph_path.exists():
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        champion = graph.get("champion", {})
        summaries["graph_champion"] = {
            "model_name": champion.get("model_name"),
            "test_macro_f1": champion.get("test_macro_f1"),
        }

    return summaries


def train_hybrid_model(
    data_path: str = "data/processed/ddi_dataset.csv",
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    max_tfidf_features: int = 20000,
    llm_model_name: str = "llama3.1:8b",
    max_llm_calls_train: int = 160,
    max_llm_calls_val: int = 80,
    max_llm_calls_test: int = 80,
) -> Dict[str, Any]:
    rows = read_ddi_dataset(data_path)
    if len(rows) < 20:
        raise ValueError(f"Not enough rows for hybrid training: {len(rows)}")

    labels = np.array([row.label for row in rows], dtype=np.int64)
    groups = np.array(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in rows], dtype=object)

    train_val_idx, test_idx = _group_stratified_split(
        y=labels,
        groups=groups,
        random_state=random_state,
        n_splits=max(2, int(round(1.0 / test_size))),
    )
    train_val_rows = _rows_by_indices(rows, train_val_idx)
    y_train_val = labels[train_val_idx]
    groups_train_val = groups[train_val_idx]

    train_idx_rel, val_idx_rel = _group_stratified_split(
        y=y_train_val,
        groups=groups_train_val,
        random_state=random_state + 1,
        n_splits=max(2, int(round(1.0 / val_size))),
    )

    train_rows = _rows_by_indices(train_val_rows, train_idx_rel)
    val_rows = _rows_by_indices(train_val_rows, val_idx_rel)
    test_rows = _rows_by_indices(rows, test_idx)

    x_train_text, y_train, text_pipeline = build_feature_pipeline(
        rows=train_rows,
        max_tfidf_features=max_tfidf_features,
    )
    x_val_text, y_val = transform_rows(val_rows, text_pipeline)
    x_test_text, y_test = transform_rows(test_rows, text_pipeline)

    text_baseline = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=random_state,
    )
    text_baseline.fit(x_train_text, y_train)
    train_text_probs = text_baseline.predict_proba(x_train_text)[:, 1]
    val_text_probs = text_baseline.predict_proba(x_val_text)[:, 1]
    test_text_probs = text_baseline.predict_proba(x_test_text)[:, 1]

    graph_context = _build_graph_context()
    llm_cache_path = Path("artifacts/features/llm_pair_predictions.json")

    x_train, train_dense, train_llm_stats = _build_hybrid_matrix(
        rows=train_rows,
        x_text=x_train_text,
        graph_context=graph_context,
        llm_cache_path=llm_cache_path,
        max_llm_calls=max_llm_calls_train,
        llm_model_name=llm_model_name,
        priority_probabilities=train_text_probs,
    )
    x_val, val_dense, val_llm_stats = _build_hybrid_matrix(
        rows=val_rows,
        x_text=x_val_text,
        graph_context=graph_context,
        llm_cache_path=llm_cache_path,
        max_llm_calls=max_llm_calls_val,
        llm_model_name=llm_model_name,
        priority_probabilities=val_text_probs,
    )
    x_test, test_dense, test_llm_stats = _build_hybrid_matrix(
        rows=test_rows,
        x_text=x_test_text,
        graph_context=graph_context,
        llm_cache_path=llm_cache_path,
        max_llm_calls=max_llm_calls_test,
        llm_model_name=llm_model_name,
        priority_probabilities=test_text_probs,
    )

    candidate_params = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 2},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 3},
        {"n_estimators": 700, "learning_rate": 0.02, "max_depth": 7, "min_child_weight": 4},
    ]

    best_model: XGBClassifier | None = None
    best_params: Dict[str, Any] | None = None
    best_val_metrics: Dict[str, Any] | None = None
    best_objective = -1.0

    for params in candidate_params:
        model = _fit_xgb_with_params(x_train, y_train, random_state, params)
        y_val_prob = model.predict_proba(x_val)[:, 1]
        val_metrics = _find_best_threshold(y_val, y_val_prob)
        objective = _threshold_objective(val_metrics)
        if best_model is None or objective > best_objective:
            best_model = model
            best_params = params
            best_val_metrics = val_metrics
            best_objective = objective

    if best_model is None or best_params is None or best_val_metrics is None:
        raise RuntimeError("Hybrid model training failed")

    y_test_prob = best_model.predict_proba(x_test)[:, 1]
    test_metrics = _compute_metrics(y_test, y_test_prob, best_val_metrics["threshold"])

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / "hybrid_xgb.joblib")
    joblib.dump(
        {
            "text_pipeline": text_pipeline,
            "graph_context": graph_context,
            "hybrid_dense_feature_names": [
                "graph_degree_1",
                "graph_degree_2",
                "graph_degree_sum",
                "graph_common_neighbors",
                "graph_jaccard",
                "graph_support_count",
                "llm_label_ddi",
                "llm_confidence",
                "llm_signed_score",
            ],
            "threshold": best_val_metrics["threshold"],
            "llm_cache_path": str(llm_cache_path),
        },
        models_dir / "hybrid_feature_bundle.joblib",
    )

    comparison = _load_existing_model_scores()
    comparison["hybrid_xgb"] = {
        "model_name": "hybrid_xgb",
        "test_macro_f1": test_metrics["macro_f1"],
    }

    metrics = {
        "config": {
            "data_path": data_path,
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "max_tfidf_features": max_tfidf_features,
            "llm_model_name": llm_model_name,
            "llm_calls": {
                "train": max_llm_calls_train,
                "val": max_llm_calls_val,
                "test": max_llm_calls_test,
            },
            "threshold_selection_metric": "0.7*macro_f1 + 0.3*recall_class_0",
        },
        "dataset": {
            "total_rows": len(rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_class_0": int((y_train == 0).sum()),
            "train_class_1": int((y_train == 1).sum()),
            "val_class_0": int((y_val == 0).sum()),
            "val_class_1": int((y_val == 1).sum()),
            "test_class_0": int((y_test == 0).sum()),
            "test_class_1": int((y_test == 1).sum()),
            "x_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "text_feature_count": len(text_pipeline.feature_names),
            "hybrid_dense_feature_count": int(train_dense.shape[1]),
        },
        "llm_usage": {
            "train": train_llm_stats,
            "val": val_llm_stats,
            "test": test_llm_stats,
        },
        "text_baseline": {
            "train_prob_mean": float(np.mean(train_text_probs)),
            "val_prob_mean": float(np.mean(val_text_probs)),
            "test_prob_mean": float(np.mean(test_text_probs)),
        },
        "hybrid_model": {
            "best_params": best_params,
            "selection_objective": float(best_objective),
            "validation": best_val_metrics,
            "test": test_metrics,
        },
        "comparison": comparison,
    }

    metrics_path = metrics_dir / "hybrid_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Saved hybrid model to %s", models_dir / "hybrid_xgb.joblib")
    logger.info("Saved hybrid feature bundle to %s", models_dir / "hybrid_feature_bundle.joblib")
    logger.info("Saved hybrid metrics to %s", metrics_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 7 hybrid DDI model")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--max-tfidf-features", type=int, default=20000)
    parser.add_argument("--llm-model-name", default="llama3.1:8b")
    parser.add_argument("--max-llm-calls-train", type=int, default=80)
    parser.add_argument("--max-llm-calls-val", type=int, default=40)
    parser.add_argument("--max-llm-calls-test", type=int, default=40)
    args = parser.parse_args()

    metrics = train_hybrid_model(
        data_path=args.data_path,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        max_tfidf_features=args.max_tfidf_features,
        llm_model_name=args.llm_model_name,
        max_llm_calls_train=args.max_llm_calls_train,
        max_llm_calls_val=args.max_llm_calls_val,
        max_llm_calls_test=args.max_llm_calls_test,
    )

    test_macro_f1 = metrics["hybrid_model"]["test"]["macro_f1"]
    logger.info("Hybrid XGBoost test macro-F1: %.4f", test_macro_f1)


if __name__ == "__main__":
    main()
