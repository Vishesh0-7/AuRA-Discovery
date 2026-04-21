"""Stage 6 Hybrid Model.

Implements two approaches:
1) Early fusion: XGBoost trained on combined TF-IDF + graph + LLM-confidence features.
2) Late fusion: final_score = 0.5*ML + 0.3*Graph + 0.2*LLM.

Outputs:
- artifacts/models/stage6_hybrid_xgb.joblib
- artifacts/models/stage6_late_fusion_bundle.joblib
- artifacts/metrics/stage6_hybrid_metrics.json
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.agents.predictor import predict_llm_safe
from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import fetch_graph_rows
from src.features.pipeline import DDIRow, read_ddi_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    return tuple(sorted((left, right)))


def _hash_row_key(drug1: str, drug2: str, text: str) -> str:
    payload = "||".join([drug1.strip().lower(), drug2.strip().lower(), text.strip().lower()])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _group_split(rows: Sequence[DDIRow], test_size: float, random_state: int) -> Tuple[List[DDIRow], List[DDIRow]]:
    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    groups = np.asarray(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in rows], dtype=object)
    n_splits = max(2, int(round(1.0 / max(test_size, 1e-6))))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy_x = np.zeros(len(rows), dtype=np.int8)
    train_idx, test_idx = next(splitter.split(dummy_x, labels, groups))

    train_rows = [rows[int(index)] for index in train_idx]
    test_rows = [rows[int(index)] for index in test_idx]
    return train_rows, test_rows


def _build_component_map(neighbor_map: Dict[str, set[str]]) -> Dict[str, int]:
    component: Dict[str, int] = {}
    component_id = 0

    for node in neighbor_map:
        if node in component:
            continue
        stack = [node]
        component[node] = component_id
        while stack:
            current = stack.pop()
            for neighbor in neighbor_map.get(current, set()):
                if neighbor not in component:
                    component[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1

    return component


def _build_graph_context(train_rows: Sequence[DDIRow]) -> Dict[str, Any]:
    train_pairs = {_canonical_pair(row.drug1, row.drug2) for row in train_rows}

    with ResearchGraph() as graph:
        graph_rows = fetch_graph_rows(graph)

    neighbor_map: Dict[str, set[str]] = {}
    for row in graph_rows:
        pair = _canonical_pair(row.drug1, row.drug2)
        if pair not in train_pairs:
            continue
        neighbor_map.setdefault(pair[0], set()).add(pair[1])
        neighbor_map.setdefault(pair[1], set()).add(pair[0])

    component_map = _build_component_map(neighbor_map)
    node_count = max(1.0, float(len(neighbor_map)))

    return {
        "neighbor_map": neighbor_map,
        "component_map": component_map,
        "node_count": node_count,
    }


def _graph_features_for_pair(drug1: str, drug2: str, graph_context: Dict[str, Any]) -> np.ndarray:
    pair = _canonical_pair(drug1, drug2)
    d1, d2 = pair

    neighbor_map: Dict[str, set[str]] = graph_context["neighbor_map"]
    component_map: Dict[str, int] = graph_context["component_map"]
    node_count = graph_context["node_count"]

    n1 = set(neighbor_map.get(d1, set())) - {d2}
    n2 = set(neighbor_map.get(d2, set())) - {d1}

    degree_1 = float(len(n1))
    degree_2 = float(len(n2))
    degree_centrality_1 = degree_1 / max(1.0, node_count - 1.0)
    degree_centrality_2 = degree_2 / max(1.0, node_count - 1.0)
    degree_centrality_sum = degree_centrality_1 + degree_centrality_2
    common_neighbors = float(len(n1 & n2))
    union_size = float(len(n1 | n2))
    jaccard = common_neighbors / union_size if union_size else 0.0
    path_existence = float(
        d1 in component_map and d2 in component_map and component_map[d1] == component_map[d2]
    )

    return np.asarray(
        [
            common_neighbors,
            degree_centrality_1,
            degree_centrality_2,
            degree_centrality_sum,
            jaccard,
            path_existence,
        ],
        dtype=np.float32,
    )


def _load_llm_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_llm_cache(cache_path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _positive_confidence(label: str, confidence: float) -> float:
    normalized = (label or "").strip().lower()
    if normalized == "ddi":
        return float(confidence)
    if normalized == "no ddi":
        return float(1.0 - confidence)
    return float(confidence)


def _llm_confidences(
    rows: Sequence[DDIRow],
    cache_path: Path,
    model_name: str,
    max_new_calls: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    cache = _load_llm_cache(cache_path)
    values: List[float] = []
    cache_hits = 0
    new_calls = 0
    fallback_count = 0

    for row in rows:
        row_key = _hash_row_key(row.drug1, row.drug2, row.text)
        if row_key in cache:
            cache_hits += 1
            cached = cache[row_key]
            values.append(_positive_confidence(str(cached.get("label", "")), float(cached.get("confidence", 0.5))))
            continue

        if new_calls < max_new_calls:
            pred = predict_llm_safe(
                drug1=row.drug1,
                drug2=row.drug2,
                text=row.text,
                model_name=model_name,
                temperature=0.0,
            )
            cache[row_key] = {"label": pred.label, "confidence": float(pred.confidence)}
            values.append(_positive_confidence(pred.label, float(pred.confidence)))
            new_calls += 1
        else:
            values.append(0.5)
            fallback_count += 1

    if new_calls > 0:
        _save_llm_cache(cache_path, cache)

    return np.asarray(values, dtype=np.float32), {
        "cache_hits": cache_hits,
        "new_calls": new_calls,
        "fallback_count": fallback_count,
        "cache_size": len(cache),
    }


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _build_early_fusion_matrix(
    x_text: csr_matrix,
    x_graph: np.ndarray,
    x_llm: np.ndarray,
    scaler: StandardScaler | None = None,
) -> Tuple[csr_matrix, StandardScaler]:
    dense = np.column_stack([x_graph, x_llm.reshape(-1, 1)]).astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        dense_scaled = scaler.fit_transform(dense)
    else:
        dense_scaled = scaler.transform(dense)
    x_combined = hstack([x_text, csr_matrix(dense_scaled)], format="csr")
    return x_combined, scaler


def run_stage6_hybrid(
    data_path: str = "data/processed/ddi_dataset.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    max_tfidf_features: int = 5000,
    llm_cache_path: str = "artifacts/features/llm_pair_predictions.json",
    llm_model_name: str = "llama3.1:8b",
    max_new_llm_calls: int = 0,
) -> Dict[str, Any]:
    rows = read_ddi_dataset(data_path)
    if len(rows) < 20:
        raise ValueError(f"Not enough rows for Stage 6 hybrid training: {len(rows)}")

    train_rows, test_rows = _group_split(rows, test_size=test_size, random_state=random_state)

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_text_train = tfidf.fit_transform([row.text for row in train_rows])
    x_text_test = tfidf.transform([row.text for row in test_rows])

    y_train = np.asarray([int(row.label) for row in train_rows], dtype=np.int64)
    y_test = np.asarray([int(row.label) for row in test_rows], dtype=np.int64)

    graph_context = _build_graph_context(train_rows)
    x_graph_train = np.asarray([_graph_features_for_pair(row.drug1, row.drug2, graph_context) for row in train_rows], dtype=np.float32)
    x_graph_test = np.asarray([_graph_features_for_pair(row.drug1, row.drug2, graph_context) for row in test_rows], dtype=np.float32)

    llm_cache = Path(llm_cache_path)
    x_llm_train, llm_train_stats = _llm_confidences(
        rows=train_rows,
        cache_path=llm_cache,
        model_name=llm_model_name,
        max_new_calls=max_new_llm_calls,
    )
    x_llm_test, llm_test_stats = _llm_confidences(
        rows=test_rows,
        cache_path=llm_cache,
        model_name=llm_model_name,
        max_new_calls=max_new_llm_calls,
    )

    x_train, early_scaler = _build_early_fusion_matrix(x_text_train, x_graph_train, x_llm_train)
    x_test, _ = _build_early_fusion_matrix(x_text_test, x_graph_test, x_llm_test, scaler=early_scaler)

    negatives = max(1, int((y_train == 0).sum()))
    positives = max(1, int((y_train == 1).sum()))
    scale_pos_weight = negatives / positives

    early_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=4,
    )
    early_model.fit(x_train, y_train)
    early_prob = early_model.predict_proba(x_test)[:, 1]

    ml_model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=random_state,
    )
    ml_model.fit(x_text_train, y_train)
    ml_prob = ml_model.predict_proba(x_text_test)[:, 1]

    graph_model = XGBClassifier(
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
    graph_model.fit(x_graph_train, y_train)
    graph_prob = graph_model.predict_proba(x_graph_test)[:, 1]

    llm_model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    llm_model.fit(x_llm_train.reshape(-1, 1), y_train)
    llm_prob = llm_model.predict_proba(x_llm_test.reshape(-1, 1))[:, 1]

    late_prob = (0.5 * ml_prob) + (0.3 * graph_prob) + (0.2 * llm_prob)

    early_metrics = _evaluate(y_test, early_prob)
    late_metrics = _evaluate(y_test, late_prob)
    ml_metrics = _evaluate(y_test, ml_prob)
    graph_metrics = _evaluate(y_test, graph_prob)
    llm_metrics = _evaluate(y_test, llm_prob)

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "tfidf_vectorizer": tfidf,
            "dense_scaler": early_scaler,
            "model": early_model,
        },
        models_dir / "stage6_hybrid_xgb.joblib",
    )
    joblib.dump(
        {
            "tfidf_vectorizer": tfidf,
            "ml_model": ml_model,
            "graph_model": graph_model,
            "llm_model": llm_model,
            "weights": {"ml": 0.5, "graph": 0.3, "llm": 0.2},
        },
        models_dir / "stage6_late_fusion_bundle.joblib",
    )

    report = {
        "config": {
            "data_path": data_path,
            "test_size": test_size,
            "random_state": random_state,
            "max_tfidf_features": max_tfidf_features,
            "llm_model_name": llm_model_name,
            "max_new_llm_calls": max_new_llm_calls,
            "scale_pos_weight": float(scale_pos_weight),
        },
        "dataset": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "test_positive": int((y_test == 1).sum()),
            "test_negative": int((y_test == 0).sum()),
            "tfidf_feature_count": int(x_text_train.shape[1]),
            "graph_feature_count": int(x_graph_train.shape[1]),
        },
        "llm_usage": {
            "train": llm_train_stats,
            "test": llm_test_stats,
        },
        "early_fusion": {
            "model": "xgboost",
            "metrics": early_metrics,
        },
        "late_fusion": {
            "formula": "0.5*ML + 0.3*Graph + 0.2*LLM",
            "metrics": late_metrics,
            "components": {
                "ml": ml_metrics,
                "graph": graph_metrics,
                "llm": llm_metrics,
            },
        },
        "comparison": {
            "winner_by_f1": "early_fusion" if early_metrics["f1"] >= late_metrics["f1"] else "late_fusion",
            "early_fusion_f1": early_metrics["f1"],
            "late_fusion_f1": late_metrics["f1"],
            "early_fusion_pr_auc": early_metrics["pr_auc"],
            "late_fusion_pr_auc": late_metrics["pr_auc"],
        },
    }

    metrics_path = metrics_dir / "stage6_hybrid_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved Stage 6 metrics to %s", metrics_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 6 hybrid approaches")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-tfidf-features", type=int, default=5000)
    parser.add_argument("--llm-cache-path", default="artifacts/features/llm_pair_predictions.json")
    parser.add_argument("--llm-model-name", default="llama3.1:8b")
    parser.add_argument("--max-new-llm-calls", type=int, default=0)
    args = parser.parse_args()

    run_stage6_hybrid(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        max_tfidf_features=args.max_tfidf_features,
        llm_cache_path=args.llm_cache_path,
        llm_model_name=args.llm_model_name,
        max_new_llm_calls=args.max_new_llm_calls,
    )


if __name__ == "__main__":
    main()