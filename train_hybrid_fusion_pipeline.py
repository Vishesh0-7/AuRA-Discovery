"""Stage-based hybrid DDI fusion pipeline.

Stage 1
    Load `text`, `graph_features`, `llm_confidence`, and `label`.
    Build TF-IDF features, normalize the numeric signals, and concatenate
    everything into a single feature matrix.

Stage 2
    Train an XGBoost classifier on the combined representation.

Stage 3
    Evaluate the model with precision, recall, F1, PR-AUC, and a
    classification report, then save metrics to `reports/hybrid_metrics.json`.

Stage 4
    Train separate TF-IDF, graph, and LLM-confidence models and combine
    their probabilities with a weighted late-fusion rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.agents.predictor import predict_llm_safe
from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import fetch_graph_rows
from src.features.pipeline import DDIRow, read_ddi_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedSplit:
    rows: List[DDIRow]
    x_full: csr_matrix
    y_full: np.ndarray
    x_text_train: csr_matrix
    x_text_test: csr_matrix
    x_train: csr_matrix
    x_test: csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray
    tfidf_vectorizer: TfidfVectorizer
    numeric_scaler: StandardScaler
    graph_train: np.ndarray
    graph_test: np.ndarray
    graph_full: np.ndarray
    llm_train: np.ndarray
    llm_test: np.ndarray
    llm_full: np.ndarray
    llm_train_stats: Dict[str, Any]
    llm_test_stats: Dict[str, Any]
    llm_full_stats: Dict[str, Any]


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    return tuple(sorted((left, right)))


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


def _group_split(rows: Sequence[DDIRow], test_size: float, random_state: int) -> Tuple[List[DDIRow], List[DDIRow]]:
    if not rows:
        raise ValueError("No rows available for splitting")

    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    groups = np.asarray(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in rows], dtype=object)
    n_splits = max(2, int(round(1.0 / max(test_size, 1e-6))))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    x_dummy = np.zeros(len(rows), dtype=np.int8)

    try:
        train_idx, test_idx = next(splitter.split(x_dummy, labels, groups))
    except ValueError:
        from sklearn.model_selection import train_test_split

        train_rows, test_rows = train_test_split(
            list(rows),
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        return list(train_rows), list(test_rows)

    train_rows = [rows[int(idx)] for idx in train_idx]
    test_rows = [rows[int(idx)] for idx in test_idx]
    return train_rows, test_rows


def _build_graph_context(train_rows: Sequence[DDIRow]) -> Dict[str, Any]:
    """Build a train-only graph neighborhood map from Neo4j rows."""

    train_pairs = {_canonical_pair(row.drug1, row.drug2) for row in train_rows}

    try:
        with ResearchGraph() as graph:
            graph_rows = fetch_graph_rows(graph)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Graph source unavailable, using zero graph features: %s", exc)
        return {"neighbor_map": {}, "support_count": {}, "available_pairs": set()}

    neighbor_map: Dict[str, set[str]] = defaultdict(set)
    support_count: Dict[Tuple[str, str], float] = {}
    available_pairs = set()

    for row in graph_rows:
        pair = _canonical_pair(row.drug1, row.drug2)
        if pair not in train_pairs:
            continue
        available_pairs.add(pair)
        neighbor_map[pair[0]].add(pair[1])
        neighbor_map[pair[1]].add(pair[0])
        support_count[pair] = float(row.support_count)

    return {
        "neighbor_map": dict(neighbor_map),
        "support_count": support_count,
        "available_pairs": available_pairs,
    }


def _graph_feature_scalar(drug1: str, drug2: str, graph_context: Dict[str, Any]) -> float:
    pair = _canonical_pair(drug1, drug2)
    d1, d2 = pair
    neighbor_map: Dict[str, set[str]] = graph_context.get("neighbor_map", {})
    support_count: Dict[Tuple[str, str], float] = graph_context.get("support_count", {})

    neighbors_1 = set(neighbor_map.get(d1, set())) - {d2}
    neighbors_2 = set(neighbor_map.get(d2, set())) - {d1}

    degree_1 = float(len(neighbors_1))
    degree_2 = float(len(neighbors_2))
    common_neighbors = neighbors_1 & neighbors_2
    common_count = float(len(common_neighbors))
    union_count = float(len(neighbors_1 | neighbors_2))
    jaccard = common_count / union_count if union_count else 0.0

    adamic_adar = 0.0
    support_sum = float(support_count.get(pair, 0.0))
    for common_node in common_neighbors:
        common_degree = len(neighbor_map.get(common_node, set()))
        if common_degree > 1:
            adamic_adar += 1.0 / math.log(common_degree)

    degree_log_sum = math.log1p(degree_1 + degree_2)
    support_log = math.log1p(support_sum)

    return (
        0.45 * common_count
        + 0.30 * jaccard
        + 0.15 * adamic_adar
        + 0.10 * degree_log_sum
        + 0.20 * support_log
    )


def _llm_positive_confidence(label: str, confidence: float) -> float:
    label_normalized = (label or "").strip().lower()
    if label_normalized == "ddi":
        return float(confidence)
    if label_normalized == "no ddi":
        return float(1.0 - confidence)
    return float(confidence)


def _llm_confidences_for_rows(
    rows: Sequence[DDIRow],
    cache_path: Path,
    model_name: str,
    max_new_calls: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cache = _load_llm_cache(cache_path)
    confidences: List[float] = []
    cache_hits = 0
    new_calls = 0
    fallback_count = 0

    for row in rows:
        row_key = _hash_row_key(row.drug1, row.drug2, row.text)
        if row_key in cache:
            cached = cache[row_key]
            cache_hits += 1
            confidences.append(_llm_positive_confidence(str(cached.get("label", "")), float(cached.get("confidence", 0.5))))
            continue

        if new_calls < max_new_calls:
            result = predict_llm_safe(
                drug1=row.drug1,
                drug2=row.drug2,
                text=row.text,
                model_name=model_name,
            )
            cache[row_key] = {
                "label": result.label,
                "confidence": float(result.confidence),
            }
            confidences.append(_llm_positive_confidence(result.label, float(result.confidence)))
            new_calls += 1
            continue

        fallback_count += 1
        confidences.append(0.5)

    if new_calls:
        _save_llm_cache(cache_path, cache)

    return np.asarray(confidences, dtype=np.float32), {
        "cache_hits": cache_hits,
        "new_calls": new_calls,
        "fallback_count": fallback_count,
        "cache_size": len(cache),
    }


def _build_feature_block(
    rows: Sequence[DDIRow],
    graph_context: Dict[str, Any],
    cache_path: Path,
    model_name: str,
    max_new_llm_calls: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    graph_values = np.asarray(
        [_graph_feature_scalar(row.drug1, row.drug2, graph_context) for row in rows],
        dtype=np.float32,
    )
    llm_values, llm_stats = _llm_confidences_for_rows(
        rows=rows,
        cache_path=cache_path,
        model_name=model_name,
        max_new_calls=max_new_llm_calls,
    )
    numeric = np.column_stack([graph_values, llm_values]).astype(np.float32)
    return graph_values, llm_values, {"numeric": numeric, "llm_stats": llm_stats}


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def _train_xgb(x_train: csr_matrix, y_train: np.ndarray, random_state: int) -> XGBClassifier:
    negative = max(1, int((y_train == 0).sum()))
    positive = max(1, int((y_train == 1).sum()))
    scale_pos_weight = negative / positive

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
    model.fit(x_train, y_train)
    return model


def _apply_smote(
    x_train: csr_matrix | np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    sampling_strategy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=3)
    if issparse(x_train):
        x_in = x_train.toarray()
    else:
        x_in = np.asarray(x_train)
    x_resampled, y_resampled = smote.fit_resample(x_in, y_train)
    return x_resampled.astype(np.float32), np.asarray(y_resampled, dtype=np.int64)


def _prepare_stage1(
    data_path: str,
    test_size: float,
    random_state: int,
    max_tfidf_features: int,
    llm_cache_path: Path,
    llm_model_name: str,
    max_new_llm_calls: int,
) -> PreparedSplit:
    rows = read_ddi_dataset(data_path)
    train_rows, test_rows = _group_split(rows, test_size=test_size, random_state=random_state)

    graph_context = _build_graph_context(train_rows)

    graph_train, llm_train, train_numeric = _build_feature_block(
        rows=train_rows,
        graph_context=graph_context,
        cache_path=llm_cache_path,
        model_name=llm_model_name,
        max_new_llm_calls=max_new_llm_calls,
    )
    graph_test, llm_test, test_numeric = _build_feature_block(
        rows=test_rows,
        graph_context=graph_context,
        cache_path=llm_cache_path,
        model_name=llm_model_name,
        max_new_llm_calls=max_new_llm_calls,
    )
    graph_full, llm_full, full_numeric = _build_feature_block(
        rows=rows,
        graph_context=graph_context,
        cache_path=llm_cache_path,
        model_name=llm_model_name,
        max_new_llm_calls=max_new_llm_calls,
    )

    texts_train = [row.text for row in train_rows]
    texts_test = [row.text for row in test_rows]
    texts_full = [row.text for row in rows]
    labels_train = np.asarray([int(row.label) for row in train_rows], dtype=np.int64)
    labels_test = np.asarray([int(row.label) for row in test_rows], dtype=np.int64)
    labels_full = np.asarray([int(row.label) for row in rows], dtype=np.int64)

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x_text_train = tfidf.fit_transform(texts_train)
    x_text_test = tfidf.transform(texts_test)
    x_text_full = tfidf.transform(texts_full)

    scaler = StandardScaler()
    x_numeric_train = scaler.fit_transform(train_numeric["numeric"])
    x_numeric_test = scaler.transform(test_numeric["numeric"])
    x_numeric_full = scaler.transform(full_numeric["numeric"])

    x_train = hstack([x_text_train, csr_matrix(x_numeric_train)], format="csr")
    x_test = hstack([x_text_test, csr_matrix(x_numeric_test)], format="csr")
    x_full = hstack([x_text_full, csr_matrix(x_numeric_full)], format="csr")

    return PreparedSplit(
        rows=rows,
        x_full=x_full,
        y_full=labels_full,
        x_text_train=x_text_train,
        x_text_test=x_text_test,
        x_train=x_train,
        x_test=x_test,
        y_train=labels_train,
        y_test=labels_test,
        tfidf_vectorizer=tfidf,
        numeric_scaler=scaler,
        graph_train=graph_train,
        graph_test=graph_test,
        graph_full=graph_full,
        llm_train=llm_train,
        llm_test=llm_test,
        llm_full=llm_full,
        llm_train_stats=train_numeric["llm_stats"],
        llm_test_stats=test_numeric["llm_stats"],
        llm_full_stats=full_numeric["llm_stats"],
    )


def _train_late_fusion_models(split: PreparedSplit, random_state: int) -> Dict[str, Any]:
    y_train = split.y_train
    y_test = split.y_test

    tfidf_model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    tfidf_model.fit(split.x_text_train, y_train)
    tfidf_prob = tfidf_model.predict_proba(split.x_text_test)[:, 1]

    graph_train = split.graph_train.reshape(-1, 1)
    graph_test = split.graph_test.reshape(-1, 1)
    graph_negative = max(1, int((y_train == 0).sum()))
    graph_positive = max(1, int((y_train == 1).sum()))
    graph_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        scale_pos_weight=graph_negative / graph_positive,
        random_state=random_state,
        n_jobs=4,
    )
    graph_model.fit(graph_train, y_train)
    graph_prob = graph_model.predict_proba(graph_test)[:, 1]

    llm_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )
    llm_model.fit(split.llm_train.reshape(-1, 1), y_train)
    llm_prob = llm_model.predict_proba(split.llm_test.reshape(-1, 1))[:, 1]

    fusion_prob = (0.5 * tfidf_prob) + (0.3 * graph_prob) + (0.2 * llm_prob)

    return {
        "tfidf_model": tfidf_model,
        "graph_model": graph_model,
        "llm_model": llm_model,
        "tfidf_prob": tfidf_prob,
        "graph_prob": graph_prob,
        "llm_prob": llm_prob,
        "fusion_prob": fusion_prob,
        "fusion_metrics": _evaluate(y_test, fusion_prob),
        "tfidf_metrics": _evaluate(y_test, tfidf_prob),
        "graph_metrics": _evaluate(y_test, graph_prob),
        "llm_metrics": _evaluate(y_test, llm_prob),
    }


def run_pipeline(
    data_path: str = "data/processed/ddi_dataset.csv",
    random_state: int = 42,
    test_size: float = 0.2,
    max_tfidf_features: int = 5000,
    llm_cache_path: str = "artifacts/features/llm_pair_predictions.json",
    llm_model_name: str = "llama3.1:8b",
    max_new_llm_calls: int = 0,
    use_smote: bool = False,
    smote_sampling_strategy: float = 0.6,
) -> Dict[str, Any]:
    llm_cache = Path(llm_cache_path)

    prepared = _prepare_stage1(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        max_tfidf_features=max_tfidf_features,
        llm_cache_path=llm_cache,
        llm_model_name=llm_model_name,
        max_new_llm_calls=max_new_llm_calls,
    )

    logger.info(
        "Stage 1 complete: train=%s test=%s full=%s tfidf_features=%s",
        prepared.x_train.shape,
        prepared.x_test.shape,
        prepared.x_full.shape,
        len(prepared.tfidf_vectorizer.get_feature_names_out()),
    )

    x_train_for_stage2: csr_matrix | np.ndarray = prepared.x_train
    y_train_for_stage2 = prepared.y_train
    smote_stats: Dict[str, Any] = {
        "applied": bool(use_smote),
        "sampling_strategy": float(smote_sampling_strategy),
        "train_shape_before": [int(prepared.x_train.shape[0]), int(prepared.x_train.shape[1])],
        "train_shape_after": [int(prepared.x_train.shape[0]), int(prepared.x_train.shape[1])],
        "class_distribution_before": {
            "negative": int((prepared.y_train == 0).sum()),
            "positive": int((prepared.y_train == 1).sum()),
        },
        "class_distribution_after": {
            "negative": int((prepared.y_train == 0).sum()),
            "positive": int((prepared.y_train == 1).sum()),
        },
    }

    if use_smote:
        x_train_for_stage2, y_train_for_stage2 = _apply_smote(
            x_train=prepared.x_train,
            y_train=prepared.y_train,
            random_state=random_state,
            sampling_strategy=smote_sampling_strategy,
        )
        smote_stats["train_shape_after"] = [int(x_train_for_stage2.shape[0]), int(x_train_for_stage2.shape[1])]
        smote_stats["class_distribution_after"] = {
            "negative": int((y_train_for_stage2 == 0).sum()),
            "positive": int((y_train_for_stage2 == 1).sum()),
        }

    xgb_model = _train_xgb(x_train_for_stage2, y_train_for_stage2, random_state=random_state)
    stage2_prob = xgb_model.predict_proba(prepared.x_test)[:, 1]
    stage2_pred = (stage2_prob >= 0.5).astype(np.int64)

    stage2_metrics = {
        "precision": float(precision_score(prepared.y_test, stage2_pred, zero_division=0)),
        "recall": float(recall_score(prepared.y_test, stage2_pred, zero_division=0)),
        "f1": float(f1_score(prepared.y_test, stage2_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(prepared.y_test, stage2_prob)),
        "classification_report": classification_report(
            prepared.y_test,
            stage2_pred,
            zero_division=0,
            output_dict=True,
        ),
        "predictions": stage2_pred.tolist(),
        "probabilities": stage2_prob.tolist(),
    }

    logger.info(
        "Stage 2 complete: precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        stage2_metrics["precision"],
        stage2_metrics["recall"],
        stage2_metrics["f1"],
        stage2_metrics["pr_auc"],
    )

    late_fusion = _train_late_fusion_models(prepared, random_state=random_state)

    logger.info(
        "Stage 4 complete: fusion precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        late_fusion["fusion_metrics"]["precision"],
        late_fusion["fusion_metrics"]["recall"],
        late_fusion["fusion_metrics"]["f1"],
        late_fusion["fusion_metrics"]["pr_auc"],
    )

    report = {
        "config": {
            "data_path": data_path,
            "random_state": random_state,
            "test_size": test_size,
            "max_tfidf_features": max_tfidf_features,
            "llm_cache_path": str(llm_cache),
            "llm_model_name": llm_model_name,
            "max_new_llm_calls": max_new_llm_calls,
            "use_smote": use_smote,
            "smote_sampling_strategy": smote_sampling_strategy,
        },
        "stage1": {
            "train_rows": int(prepared.x_train.shape[0]),
            "test_rows": int(prepared.x_test.shape[0]),
            "total_rows": int(len(prepared.rows)),
            "tfidf_features": int(len(prepared.tfidf_vectorizer.get_feature_names_out())),
            "numeric_features": 2,
            "llm_usage": {
                "train": prepared.llm_train_stats,
                "test": prepared.llm_test_stats,
                "full": prepared.llm_full_stats,
            },
        },
        "smote": smote_stats,
        "stage2": stage2_metrics,
        "stage4": {
            "tfidf": late_fusion["tfidf_metrics"],
            "graph": late_fusion["graph_metrics"],
            "llm": late_fusion["llm_metrics"],
            "fusion": late_fusion["fusion_metrics"],
            "weights": {
                "tfidf": 0.5,
                "graph": 0.3,
                "llm": 0.2,
            },
        },
    }

    metrics_path = Path("reports/hybrid_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)

    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "tfidf_vectorizer": prepared.tfidf_vectorizer,
            "numeric_scaler": prepared.numeric_scaler,
            "xgb_model": xgb_model,
            "late_fusion": {
                "tfidf_model": late_fusion["tfidf_model"],
                "graph_model": late_fusion["graph_model"],
                "llm_model": late_fusion["llm_model"],
            },
        },
        model_dir / "staged_hybrid_bundle.joblib",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train staged hybrid DDI fusion models")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-tfidf-features", type=int, default=5000)
    parser.add_argument("--llm-cache-path", default="artifacts/features/llm_pair_predictions.json")
    parser.add_argument("--llm-model-name", default="llama3.1:8b")
    parser.add_argument("--max-new-llm-calls", type=int, default=0)
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--smote-sampling-strategy", type=float, default=0.6)
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        random_state=args.random_state,
        test_size=args.test_size,
        max_tfidf_features=args.max_tfidf_features,
        llm_cache_path=args.llm_cache_path,
        llm_model_name=args.llm_model_name,
        max_new_llm_calls=args.max_new_llm_calls,
        use_smote=args.use_smote,
        smote_sampling_strategy=args.smote_sampling_strategy,
    )


if __name__ == "__main__":
    main()