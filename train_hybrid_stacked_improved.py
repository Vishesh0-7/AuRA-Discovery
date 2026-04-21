"""Stage 2: Improve hybrid model using stacked ensemble.

This script implements:
1) Base models
   - ML model on TF-IDF reduced with TruncatedSVD
   - Graph model on improved Neo4j graph signals
   - LLM confidence-only probabilistic model
2) Stacking
   - Meta-model trained on base-model probabilities
3) Comparison
   - Old hybrid (feature concatenation / early fusion)
   - New hybrid (stacked ensemble)
4) Outputs
   - Metrics JSON/CSV/Markdown and comparison plots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.agents.predictor import predict_llm_safe
from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import (
    compute_node2vec_embeddings,
    compute_pagerank,
    fetch_graph_rows,
)
from src.features.pipeline import DDIRow, read_ddi_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    return tuple(sorted((left, right)))


def _row_key(drug1: str, drug2: str, text: str) -> str:
    payload = "||".join([drug1.strip().lower(), drug2.strip().lower(), text.strip().lower()])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _group_stratified_split(
    rows: Sequence[DDIRow],
    test_size: float,
    random_state: int,
) -> Tuple[List[DDIRow], List[DDIRow]]:
    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    groups = np.asarray(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in rows], dtype=object)
    n_splits = max(2, int(round(1.0 / max(test_size, 1e-6))))
    split = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    x_dummy = np.zeros(len(rows), dtype=np.int8)
    train_idx, test_idx = next(split.split(x_dummy, labels, groups))
    train_rows = [rows[int(i)] for i in train_idx]
    test_rows = [rows[int(i)] for i in test_idx]
    return train_rows, test_rows


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _shortest_path_without_direct_edge(
    source: str,
    target: str,
    neighbor_map: Dict[str, set[str]],
    max_depth: int = 6,
) -> float:
    if source == target:
        return 0.0
    if source not in neighbor_map or target not in neighbor_map:
        return float(max_depth + 1)

    queue = deque([(source, 0)])
    visited = {source}

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = set(neighbor_map.get(node, set()))
        if node == source and target in neighbors:
            neighbors.remove(target)

        for neighbor in neighbors:
            if neighbor == target:
                return float(depth + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return float(max_depth + 1)


def _build_graph_context(train_rows: Sequence[DDIRow]) -> Dict[str, Any]:
    train_pairs = {_canonical_pair(row.drug1, row.drug2) for row in train_rows}

    with ResearchGraph() as graph:
        graph_rows = fetch_graph_rows(graph)

    neighbor_map: Dict[str, set[str]] = defaultdict(set)
    support_count: Dict[Tuple[str, str], float] = {}

    for row in graph_rows:
        pair = _canonical_pair(row.drug1, row.drug2)
        if pair not in train_pairs:
            continue
        d1, d2 = pair
        neighbor_map[d1].add(d2)
        neighbor_map[d2].add(d1)
        support_count[pair] = float(row.support_count)

    pagerank = compute_pagerank(dict(neighbor_map))
    node2vec = compute_node2vec_embeddings(dict(neighbor_map), embedding_dim=16)

    return {
        "neighbor_map": dict(neighbor_map),
        "support_count": support_count,
        "pagerank": pagerank,
        "node2vec": node2vec,
    }


def _graph_features(row: DDIRow, graph_context: Dict[str, Any]) -> np.ndarray:
    d1, d2 = _canonical_pair(row.drug1, row.drug2)
    neighbor_map: Dict[str, set[str]] = graph_context["neighbor_map"]
    support_count: Dict[Tuple[str, str], float] = graph_context["support_count"]
    pagerank: Dict[str, float] = graph_context["pagerank"]
    node2vec: Dict[str, np.ndarray] | None = graph_context["node2vec"]

    n1 = set(neighbor_map.get(d1, set())) - {d2}
    n2 = set(neighbor_map.get(d2, set())) - {d1}

    degree_1 = float(len(n1))
    degree_2 = float(len(n2))
    node_count = max(1.0, float(len(neighbor_map)))
    degree_centrality_1 = degree_1 / max(1.0, node_count - 1.0)
    degree_centrality_2 = degree_2 / max(1.0, node_count - 1.0)

    shared = n1 & n2
    shared_neighbors = float(len(shared))
    union_size = float(len(n1 | n2))
    jaccard = shared_neighbors / union_size if union_size else 0.0

    shortest_path = _shortest_path_without_direct_edge(d1, d2, neighbor_map)
    shortest_path_inv = 1.0 / (1.0 + shortest_path)

    pr1 = float(pagerank.get(d1, 0.0))
    pr2 = float(pagerank.get(d2, 0.0))

    support = float(support_count.get((d1, d2), 0.0))

    n2v_cos = 0.0
    n2v_l2 = 0.0
    if node2vec is not None:
        v1 = node2vec.get(d1)
        v2 = node2vec.get(d2)
        if v1 is not None and v2 is not None:
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom > 0.0:
                n2v_cos = float(np.dot(v1, v2) / denom)
            n2v_l2 = float(np.linalg.norm(v1 - v2))

    return np.asarray(
        [
            degree_centrality_1,
            degree_centrality_2,
            shortest_path,
            shortest_path_inv,
            shared_neighbors,
            jaccard,
            pr1,
            pr2,
            pr1 + pr2,
            abs(pr1 - pr2),
            support,
            math.log1p(support),
            n2v_cos,
            n2v_l2,
        ],
        dtype=np.float32,
    )


def _load_llm_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_llm_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _llm_positive_confidence(label: str, confidence: float) -> float:
    normalized = (label or "").strip().lower()
    if normalized == "ddi":
        return float(confidence)
    if normalized == "no ddi":
        return float(1.0 - confidence)
    return float(confidence)


def _llm_confidence_vector(
    rows: Sequence[DDIRow],
    cache_path: Path,
    model_name: str,
    max_new_calls: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    cache = _load_llm_cache(cache_path)
    values: List[float] = []
    hits = 0
    new_calls = 0
    fallback = 0

    for row in rows:
        key = _row_key(row.drug1, row.drug2, row.text)
        if key in cache:
            hits += 1
            cached = cache[key]
            values.append(_llm_positive_confidence(str(cached.get("label", "")), float(cached.get("confidence", 0.5))))
            continue

        if new_calls < max_new_calls:
            pred = predict_llm_safe(
                drug1=row.drug1,
                drug2=row.drug2,
                text=row.text,
                model_name=model_name,
                temperature=0.0,
            )
            cache[key] = {"label": pred.label, "confidence": float(pred.confidence)}
            values.append(_llm_positive_confidence(pred.label, float(pred.confidence)))
            new_calls += 1
        else:
            values.append(0.5)
            fallback += 1

    if new_calls > 0:
        _save_llm_cache(cache_path, cache)

    return np.asarray(values, dtype=np.float32), {
        "cache_hits": hits,
        "new_calls": new_calls,
        "fallback_count": fallback,
        "cache_size": len(cache),
    }


def run_improved_hybrid_stacking(
    data_path: str = "data/processed/ddi_dataset.csv",
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    max_tfidf_features: int = 5000,
    svd_components: int = 300,
    llm_model_name: str = "llama3.1:8b",
    llm_cache_path: str = "artifacts/features/llm_pair_predictions.json",
    max_new_llm_calls: int = 0,
) -> Dict[str, Any]:
    rows = read_ddi_dataset(data_path)
    if len(rows) < 30:
        raise ValueError(f"Not enough rows for stacking experiment: {len(rows)}")

    train_val_rows, test_rows = _group_stratified_split(rows, test_size=test_size, random_state=random_state)
    train_rows, val_rows = _group_stratified_split(train_val_rows, test_size=val_size, random_state=random_state + 1)

    y_train = np.asarray([int(row.label) for row in train_rows], dtype=np.int64)
    y_val = np.asarray([int(row.label) for row in val_rows], dtype=np.int64)
    y_test = np.asarray([int(row.label) for row in test_rows], dtype=np.int64)

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_train_tfidf = tfidf.fit_transform([row.text for row in train_rows])
    x_val_tfidf = tfidf.transform([row.text for row in val_rows])
    x_test_tfidf = tfidf.transform([row.text for row in test_rows])

    max_components = max(2, min(svd_components, x_train_tfidf.shape[1] - 1, x_train_tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    x_train_svd = svd.fit_transform(x_train_tfidf)
    x_val_svd = svd.transform(x_val_tfidf)
    x_test_svd = svd.transform(x_test_tfidf)

    svd_scaler = StandardScaler()
    x_train_svd = svd_scaler.fit_transform(x_train_svd)
    x_val_svd = svd_scaler.transform(x_val_svd)
    x_test_svd = svd_scaler.transform(x_test_svd)

    graph_context = _build_graph_context(train_rows)
    x_train_graph = np.asarray([_graph_features(row, graph_context) for row in train_rows], dtype=np.float32)
    x_val_graph = np.asarray([_graph_features(row, graph_context) for row in val_rows], dtype=np.float32)
    x_test_graph = np.asarray([_graph_features(row, graph_context) for row in test_rows], dtype=np.float32)

    graph_scaler = StandardScaler()
    x_train_graph = graph_scaler.fit_transform(x_train_graph)
    x_val_graph = graph_scaler.transform(x_val_graph)
    x_test_graph = graph_scaler.transform(x_test_graph)

    llm_cache = Path(llm_cache_path)
    x_train_llm, llm_train_stats = _llm_confidence_vector(train_rows, llm_cache, llm_model_name, max_new_llm_calls)
    x_val_llm, llm_val_stats = _llm_confidence_vector(val_rows, llm_cache, llm_model_name, max_new_llm_calls)
    x_test_llm, llm_test_stats = _llm_confidence_vector(test_rows, llm_cache, llm_model_name, max_new_llm_calls)

    ml_model = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=3000, random_state=random_state)
    ml_model.fit(x_train_svd, y_train)
    val_ml_prob = ml_model.predict_proba(x_val_svd)[:, 1]
    test_ml_prob = ml_model.predict_proba(x_test_svd)[:, 1]

    neg = max(1, int((y_train == 0).sum()))
    pos = max(1, int((y_train == 1).sum()))
    graph_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        random_state=random_state,
        n_jobs=4,
    )
    graph_model.fit(x_train_graph, y_train)
    val_graph_prob = graph_model.predict_proba(x_val_graph)[:, 1]
    test_graph_prob = graph_model.predict_proba(x_test_graph)[:, 1]

    llm_model = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=random_state)
    llm_model.fit(x_train_llm.reshape(-1, 1), y_train)
    val_llm_prob = llm_model.predict_proba(x_val_llm.reshape(-1, 1))[:, 1]
    test_llm_prob = llm_model.predict_proba(x_test_llm.reshape(-1, 1))[:, 1]

    meta_train = np.column_stack([val_ml_prob, val_graph_prob, val_llm_prob]).astype(np.float32)
    meta_test = np.column_stack([test_ml_prob, test_graph_prob, test_llm_prob]).astype(np.float32)

    meta_model = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=random_state)
    meta_model.fit(meta_train, y_val)
    stacked_test_prob = meta_model.predict_proba(meta_test)[:, 1]

    metrics = {
        "ml_base": _evaluate(y_test, test_ml_prob),
        "graph_base_improved": _evaluate(y_test, test_graph_prob),
        "llm_base": _evaluate(y_test, test_llm_prob),
        "hybrid_stacked": _evaluate(y_test, stacked_test_prob),
    }

    old_hybrid_path = Path("artifacts/metrics/stage6_hybrid_metrics.json")
    old_hybrid = {}
    if old_hybrid_path.exists():
        old_report = json.loads(old_hybrid_path.read_text(encoding="utf-8"))
        old_hybrid = {
            "precision": float(old_report.get("early_fusion", {}).get("metrics", {}).get("precision", 0.0)),
            "recall": float(old_report.get("early_fusion", {}).get("metrics", {}).get("recall", 0.0)),
            "f1": float(old_report.get("early_fusion", {}).get("metrics", {}).get("f1", 0.0)),
            "pr_auc": float(old_report.get("early_fusion", {}).get("metrics", {}).get("pr_auc", 0.0)),
        }

    graph_baseline = {}
    baseline_path = Path("artifacts/metrics/graph_metrics.json")
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
        champion = str(base.get("champion", {}).get("model_name", "graph_xgb"))
        test_block = base.get("models", {}).get(champion, {}).get("test", {})
        graph_baseline = {
            "precision": float(test_block.get("precision_class_1", 0.0)),
            "recall": float(test_block.get("recall_class_1", 0.0)),
            "f1": float(test_block.get("f1_class_1", 0.0)),
            "pr_auc": float(test_block.get("pr_auc", 0.0)),
        }

    comparison_rows = [
        {"model": "Old Hybrid (Feature Concat)", **old_hybrid},
        {"model": "New Hybrid (Stacked)", **metrics["hybrid_stacked"]},
    ] if old_hybrid else [
        {"model": "New Hybrid (Stacked)", **metrics["hybrid_stacked"]},
    ]

    graph_improvement = {}
    if graph_baseline:
        graph_improvement = {
            "baseline_graph_f1": float(graph_baseline["f1"]),
            "improved_graph_f1": float(metrics["graph_base_improved"]["f1"]),
            "delta_graph_f1": float(metrics["graph_base_improved"]["f1"] - graph_baseline["f1"]),
            "baseline_graph_pr_auc": float(graph_baseline["pr_auc"]),
            "improved_graph_pr_auc": float(metrics["graph_base_improved"]["pr_auc"]),
            "delta_graph_pr_auc": float(metrics["graph_base_improved"]["pr_auc"] - graph_baseline["pr_auc"]),
        }

    report = {
        "config": {
            "data_path": data_path,
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "max_tfidf_features": max_tfidf_features,
            "svd_components_used": int(max_components),
            "llm_model_name": llm_model_name,
            "max_new_llm_calls": max_new_llm_calls,
            "node2vec_enabled": bool(graph_context.get("node2vec") is not None),
        },
        "dataset": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_class_0": int((y_train == 0).sum()),
            "train_class_1": int((y_train == 1).sum()),
            "val_class_0": int((y_val == 0).sum()),
            "val_class_1": int((y_val == 1).sum()),
            "test_class_0": int((y_test == 0).sum()),
            "test_class_1": int((y_test == 1).sum()),
        },
        "llm_usage": {
            "train": llm_train_stats,
            "val": llm_val_stats,
            "test": llm_test_stats,
        },
        "metrics": metrics,
        "comparison_old_vs_new_hybrid": comparison_rows,
        "graph_feature_enhancement_impact": graph_improvement,
    }

    metrics_dir = Path("artifacts/metrics")
    models_dir = Path("artifacts/models")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "hybrid_stacked_improved_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    joblib.dump(
        {
            "tfidf_vectorizer": tfidf,
            "svd": svd,
            "svd_scaler": svd_scaler,
            "graph_scaler": graph_scaler,
            "ml_model": ml_model,
            "graph_model": graph_model,
            "llm_model": llm_model,
            "meta_model": meta_model,
        },
        models_dir / "hybrid_stacked_improved_bundle.joblib",
    )

    csv_path = reports_dir / "hybrid_stacked_comparison.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("model,precision,recall,f1,pr_auc\n")
        for row in comparison_rows:
            handle.write(
                f"{row['model']},{row.get('precision', 0.0):.6f},{row.get('recall', 0.0):.6f},{row.get('f1', 0.0):.6f},{row.get('pr_auc', 0.0):.6f}\n"
            )

    md_path = reports_dir / "hybrid_stacked_comparison.md"
    lines = [
        "| Model | Precision | Recall | F1 | PR-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['model']} | {row.get('precision', 0.0):.4f} | {row.get('recall', 0.0):.4f} | {row.get('f1', 0.0):.4f} | {row.get('pr_auc', 0.0):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(comparison_rows) >= 2:
        labels = [row["model"] for row in comparison_rows]
        f1_values = [row.get("f1", 0.0) for row in comparison_rows]
        pr_values = [row.get("pr_auc", 0.0) for row in comparison_rows]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, f1_values, width=width, label="F1", color="#4C78A8")
        ax.bar(x + width / 2, pr_values, width=width, label="PR-AUC", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Old vs New Hybrid")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / "hybrid_old_vs_stacked.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if graph_improvement:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["Graph Baseline", "Improved Graph Features"]
        f1_vals = [graph_improvement["baseline_graph_f1"], graph_improvement["improved_graph_f1"]]
        ax.bar(labels, f1_vals, color=["#B279A2", "#54A24B"])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("F1")
        ax.set_title("Graph Feature Enhancement Impact")
        ax.grid(axis="y", alpha=0.2)
        for idx, value in enumerate(f1_vals):
            ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(figures_dir / "graph_enhancement_impact.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved improved stacked hybrid metrics to %s", metrics_dir / "hybrid_stacked_improved_metrics.json")
    logger.info("Saved comparison table to %s and %s", csv_path, md_path)
    logger.info("Saved model bundle to %s", models_dir / "hybrid_stacked_improved_bundle.joblib")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved stacked hybrid model")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--max-tfidf-features", type=int, default=5000)
    parser.add_argument("--svd-components", type=int, default=300)
    parser.add_argument("--llm-model-name", default="llama3.1:8b")
    parser.add_argument("--llm-cache-path", default="artifacts/features/llm_pair_predictions.json")
    parser.add_argument("--max-new-llm-calls", type=int, default=0)
    args = parser.parse_args()

    run_improved_hybrid_stacking(
        data_path=args.data_path,
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        max_tfidf_features=args.max_tfidf_features,
        svd_components=args.svd_components,
        llm_model_name=args.llm_model_name,
        llm_cache_path=args.llm_cache_path,
        max_new_llm_calls=args.max_new_llm_calls,
    )


if __name__ == "__main__":
    main()