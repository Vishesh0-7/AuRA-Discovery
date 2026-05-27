"""Research-grade inference service for pre-trained DDI models."""

from __future__ import annotations

import json
import logging
import math
import re
import html
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack

from src.agents.predictor import predict_llm_stage4_safe
from src.features.graph_pipeline import (
    GraphDDIRow,
    _shortest_path_length_without_direct_edge,
    build_component_map,
    compute_node2vec_embeddings,
    compute_pagerank,
    split_rows,
)
from src.features.pipeline import DDIRow, FeaturePipeline, read_ddi_dataset, transform_rows
from train_graph_gnn_model import GraphDDIGNN


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "models"
DATA_PATH = ROOT / "data" / "processed" / "ddi_dataset.csv"


RISK_KEYWORDS = {
    "severe": [
        "life-threatening",
        "fatal",
        "contraindicated",
        "major toxicity",
        "toxicity",
        "black box",
        "arrhythmia",
        "torsades",
        "qt prolong",
        "hemorrhage",
    ],
    "moderate": [
        "monitor",
        "dose adjustment",
        "adjust dose",
        "caution",
        "increased exposure",
        "elevated concentration",
        "requires monitoring",
    ],
    "mild": [
        "minor",
        "no significant",
        "minimal",
        "weak interaction",
        "not clinically significant",
    ],
}

TYPE_PATTERNS = [
    ("Pharmacokinetic - CYP3A4 inhibition", ["cyp3a4", "metabolism inhibition", "inhibits metabolism"]),
    ("Pharmacokinetic - CYP induction", ["cyp induction", "induces metabolism", "enzyme induction"]),
    ("Pharmacodynamic - QT prolongation", ["qt prolong", "torsades", "arrhythmia"]),
    ("Pharmacodynamic - Bleeding risk", ["bleeding", "hemorrhage", "anticoagulant effect"]),
    ("Pharmacodynamic - CNS depression", ["sedation", "cns depression", "respiratory depression"]),
    ("Pharmacokinetic - Increased plasma concentration", ["increases plasma", "increased concentration", "elevated levels"]),
    ("No clinically meaningful interaction", ["no significant interaction", "no clinically significant interaction"]),
]

MODEL_WEIGHTS = {
    "LLM Structured Predictor": 0.30,
    "Graph-based Neo4j XGBoost": 0.25,
    "Graph GNN (GraphSAGE)": 0.20,
    "XGBoost": 0.15,
    "Logistic Regression": 0.05,
    "TF-IDF + Pair Champion": 0.05,
}

SEVERITY_SCORES = {
    "Mild": 1.0,
    "Moderate": 2.0,
    "Severe": 3.0,
}

SEVERITY_BUCKET_TO_LABEL = {
    "Mild": "Safe",
    "Moderate": "Caution",
    "Severe": "Dangerous",
}


@dataclass(frozen=True)
class ModelOutput:
    """One model inference output for API and UI rendering."""

    model: str
    prediction: str
    confidence: float | None
    interaction_type: str
    severity: str
    severity_label: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        confidence_pct = None if self.confidence is None else max(0.0, min(100.0, self.confidence * 100.0))
        return {
            "model": self.model,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "confidence_pct": confidence_pct,
            "confidence_label": "n/a" if confidence_pct is None else f"{confidence_pct:.1f}%",
            "interaction_type": self.interaction_type,
            "severity": self.severity,
            "severity_label": self.severity_label,
            "notes": self.notes,
            "prediction_class": "positive" if self.prediction == "DDI" else ("negative" if self.prediction == "No DDI" else "neutral"),
            "severity_class": self.severity_label.lower(),
        }


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    return tuple(sorted((left, right)))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_model_artifact(artifact: Any, fallback_threshold: float = 0.5) -> Tuple[Any, float]:
    """Return (estimator, threshold) from either plain model or serialized bundle dict."""
    if isinstance(artifact, dict):
        model = artifact.get("model")
        threshold = float(artifact.get("threshold", fallback_threshold))
        if model is None:
            raise ValueError("Serialized model bundle is missing 'model'")
        return model, threshold
    return artifact, fallback_threshold


def _evaluate_prediction(probability: float, threshold: float) -> str:
    return "DDI" if probability >= threshold else "No DDI"


def _normalize_severity_label(severity: str) -> str:
    mapping = {"Severe": "Dangerous", "Moderate": "Caution", "Mild": "Safe"}
    return mapping.get(severity, "Safe")


def _split_sentences(text: str) -> List[str]:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]


def _highlight_risk_phrases(sentence: str) -> str:
    escaped = html.escape(sentence)
    phrases: List[str] = []
    for words in RISK_KEYWORDS.values():
        phrases.extend(words)
    phrases.extend(["interaction", "increases plasma concentration", "no significant interaction"])
    unique_phrases = sorted(set(phrases), key=len, reverse=True)

    highlighted = escaped
    for phrase in unique_phrases:
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted)
    return highlighted


def _extract_evidence_sentences(drug1: str, drug2: str, abstract: str, top_k: int = 3) -> List[str]:
    text = (abstract or "").strip()
    if not text:
        return []

    d1 = drug1.strip().lower()
    d2 = drug2.strip().lower()
    sentences = _split_sentences(text)
    scored: List[Tuple[int, str]] = []

    for sentence in sentences:
        s = sentence.lower()
        score = 0
        has_d1 = d1 in s
        has_d2 = d2 in s
        if has_d1 and has_d2:
            score += 4
        elif has_d1 or has_d2:
            score += 1

        for severity, words in RISK_KEYWORDS.items():
            for word in words:
                if word in s:
                    if severity == "severe":
                        score += 3
                    elif severity == "moderate":
                        score += 2
                    else:
                        score += 1
        if "interaction" in s:
            score += 2
        if score > 0:
            scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [_highlight_risk_phrases(sentence) for _, sentence in scored[:top_k]]


def _infer_interaction_type(text: str, prediction: str) -> str:
    if prediction != "DDI":
        return "No clinically meaningful interaction"

    lower = (text or "").lower()
    for interaction_type, keywords in TYPE_PATTERNS:
        if any(keyword in lower for keyword in keywords):
            return interaction_type
    return "General pharmacological interaction"


def _infer_severity(text: str, prediction: str, confidence: float) -> Tuple[str, str]:
    if prediction != "DDI":
        return "Mild", _normalize_severity_label("Mild")

    lower = (text or "").lower()
    if any(keyword in lower for keyword in RISK_KEYWORDS["severe"]):
        severity = "Severe"
    elif any(keyword in lower for keyword in RISK_KEYWORDS["moderate"]):
        severity = "Moderate"
    elif any(keyword in lower for keyword in RISK_KEYWORDS["mild"]):
        severity = "Mild"
    elif confidence >= 0.85:
        severity = "Severe"
    elif confidence >= 0.65:
        severity = "Moderate"
    else:
        severity = "Mild"
    return severity, _normalize_severity_label(severity)


def _severity_bucket_from_score(score: float) -> Tuple[str, str]:
    if score <= 1.5:
        severity = "Mild"
    elif score <= 2.3:
        severity = "Moderate"
    else:
        severity = "Severe"
    return severity, SEVERITY_BUCKET_TO_LABEL[severity]


def _clinical_prediction_from_score(score: float) -> str:
    if score < 0.4:
        return "No DDI"
    if score < 0.7:
        return "Mild DDI"
    if score < 0.85:
        return "Moderate DDI"
    return "Severe DDI"


def _clinical_interpretation(final_prediction: str, interaction_type: str, severity_label: str) -> str:
    if final_prediction == "No DDI" or interaction_type == "No clinically meaningful interaction":
        return "No clinically meaningful interaction despite co-occurrence in literature."
    if severity_label == "Safe":
        return "Mild interaction with minimal clinical impact."
    if severity_label == "Caution":
        return "Interaction warrants monitoring or dose adjustment."
    return "Potentially dangerous interaction requiring clinical caution."


def _model_binary_prediction(output: ModelOutput) -> int:
    return 1 if output.prediction == "DDI" else 0


def _model_severity_score(output: ModelOutput) -> float:
    if output.severity not in SEVERITY_SCORES:
        return SEVERITY_SCORES["Mild"]
    return SEVERITY_SCORES[output.severity]


def _graph_feature_map(drug1: str, drug2: str, neighbor_map: Dict[str, set[str]]) -> Dict[str, float]:
    d1, d2 = _canonical_pair(drug1, drug2)
    n1 = set(neighbor_map.get(d1, set())) - {d2}
    n2 = set(neighbor_map.get(d2, set())) - {d1}
    node_count = max(1.0, float(len(neighbor_map)))

    degree_1 = float(len(n1))
    degree_2 = float(len(n2))
    degree_centrality_1 = degree_1 / max(1.0, node_count - 1.0)
    degree_centrality_2 = degree_2 / max(1.0, node_count - 1.0)
    common_neighbors = n1 & n2
    common_count = float(len(common_neighbors))
    union_count = float(len(n1 | n2))
    jaccard = common_count / union_count if union_count else 0.0
    component_map = build_component_map(neighbor_map)
    path_existence = float(d1 in component_map and d2 in component_map and component_map.get(d1) == component_map.get(d2))
    shortest_path_length = _shortest_path_length_without_direct_edge(d1, d2, neighbor_map)
    shortest_path_inverse = 1.0 / (1.0 + shortest_path_length)

    pagerank_map = compute_pagerank(neighbor_map)
    pagerank_1 = float(pagerank_map.get(d1, 0.0))
    pagerank_2 = float(pagerank_map.get(d2, 0.0))

    node2vec_embeddings = compute_node2vec_embeddings(neighbor_map)
    node2vec_cosine = 0.0
    node2vec_l2 = 0.0
    if node2vec_embeddings is not None:
        v1 = node2vec_embeddings.get(d1)
        v2 = node2vec_embeddings.get(d2)
        if v1 is not None and v2 is not None:
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom > 0.0:
                node2vec_cosine = float(np.dot(v1, v2) / denom)
            node2vec_l2 = float(np.linalg.norm(v1 - v2))

    adamic_adar = 0.0
    resource_allocation = 0.0
    for common_node in common_neighbors:
        common_degree = len(neighbor_map.get(common_node, set()))
        if common_degree > 1:
            adamic_adar += 1.0 / math.log(common_degree)
        if common_degree > 0:
            resource_allocation += 1.0 / common_degree

    salton = common_count / math.sqrt(max(1.0, degree_1 * degree_2))
    sorensen = (2.0 * common_count) / max(1.0, degree_1 + degree_2)
    hub_promoted = common_count / max(1.0, min(degree_1, degree_2))
    hub_depressed = common_count / max(1.0, max(degree_1, degree_2))
    support_count = 1.0 if d2 in neighbor_map.get(d1, set()) else 0.0

    return {
        "degree_1": degree_1,
        "degree_2": degree_2,
        "degree_centrality_1": degree_centrality_1,
        "degree_centrality_2": degree_centrality_2,
        "degree_centrality_sum": degree_centrality_1 + degree_centrality_2,
        "degree_sum": degree_1 + degree_2,
        "degree_log_sum": math.log1p(degree_1 + degree_2),
        "degree_diff": abs(degree_1 - degree_2),
        "common_neighbors": common_count,
        "shared_neighbors": common_count,
        "common_neighbors_ratio": common_count / max(1.0, min(degree_1, degree_2)),
        "jaccard": jaccard,
        "path_existence": path_existence,
        "shortest_path_length": shortest_path_length,
        "shortest_path_inverse": shortest_path_inverse,
        "pagerank_1": pagerank_1,
        "pagerank_2": pagerank_2,
        "pagerank_sum": pagerank_1 + pagerank_2,
        "pagerank_diff": abs(pagerank_1 - pagerank_2),
        "node2vec_cosine": node2vec_cosine,
        "node2vec_l2": node2vec_l2,
        "adamic_adar": float(adamic_adar),
        "resource_allocation": float(resource_allocation),
        "salton": float(salton),
        "sorensen": float(sorensen),
        "hub_promoted": float(hub_promoted),
        "hub_depressed": float(hub_depressed),
        "union_size": union_count,
        "preferential_attachment": degree_1 * degree_2,
        "support_count": support_count,
        "support_count_log": math.log1p(support_count),
    }


def _build_csv_graph_rows(csv_path: Path) -> List[GraphDDIRow]:
    rows = read_ddi_dataset(csv_path)
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        left, right = _canonical_pair(row.drug1, row.drug2)
        if not left or not right:
            continue
        key = (left, right)
        label = int(row.label)
        if key not in aggregated:
            aggregated[key] = {
                "drug1": left,
                "drug2": right,
                "label": label,
                "support_count": 1,
                "validated_count": 1 if label == 1 else 0,
            }
        else:
            aggregated[key]["support_count"] += 1
            aggregated[key]["validated_count"] += 1 if label == 1 else 0
            aggregated[key]["label"] = max(aggregated[key]["label"], label)

    graph_rows = [GraphDDIRow(**value) for value in aggregated.values()]
    graph_rows.sort(key=lambda item: (item.drug1, item.drug2))
    return graph_rows


def _build_gnn_context(csv_path: Path) -> Dict[str, Any]:
    rows = _build_csv_graph_rows(csv_path)
    train_rows, _, _ = split_rows(rows, test_size=0.2, val_size=0.2, random_state=42)

    train_neighbor_map: Dict[str, set[str]] = {}
    for row in train_rows:
        train_neighbor_map.setdefault(row.drug1, set()).add(row.drug2)
        train_neighbor_map.setdefault(row.drug2, set()).add(row.drug1)

    node_set: set[str] = set()
    for row in rows:
        node_set.add(row.drug1)
        node_set.add(row.drug2)

    nodes = sorted(node for node in node_set if node)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    support_by_node: Dict[str, float] = {node: 0.0 for node in nodes}
    for row in train_rows:
        support_by_node[row.drug1] = support_by_node.get(row.drug1, 0.0) + float(row.support_count)
        support_by_node[row.drug2] = support_by_node.get(row.drug2, 0.0) + float(row.support_count)

    node_count = max(1.0, float(len(nodes)))
    features = np.zeros((len(nodes), 3), dtype=np.float32)
    pagerank_map = compute_pagerank(train_neighbor_map)
    for node, idx in node_to_idx.items():
        degree = float(len(train_neighbor_map.get(node, set())))
        degree_centrality = degree / max(1.0, node_count - 1.0)
        pagerank = float(pagerank_map.get(node, 0.0))
        support_log = float(np.log1p(support_by_node.get(node, 0.0)))
        features[idx] = np.array([degree_centrality, pagerank, support_log], dtype=np.float32)

    edge_pairs: List[Tuple[int, int]] = []
    for src, neighbors in train_neighbor_map.items():
        src_idx = node_to_idx.get(src)
        if src_idx is None:
            continue
        for dst in neighbors:
            dst_idx = node_to_idx.get(dst)
            if dst_idx is None:
                continue
            edge_pairs.append((src_idx, dst_idx))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return {
        "rows": rows,
        "train_rows": train_rows,
        "node_to_idx": node_to_idx,
        "node_features": torch.tensor(features, dtype=torch.float32),
        "edge_index": edge_index,
    }


def _load_stage6_graph_context(csv_path: Path) -> Dict[str, Any]:
    rows = read_ddi_dataset(csv_path)
    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    groups = np.asarray(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in rows], dtype=object)

    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=max(2, int(round(1.0 / 0.2))), shuffle=True, random_state=42)
    dummy_x = np.zeros(len(rows), dtype=np.int8)
    train_idx, _ = next(splitter.split(dummy_x, labels, groups))
    train_rows = [rows[int(index)] for index in train_idx]

    neighbor_map: Dict[str, set[str]] = {}
    support_count: Dict[Tuple[str, str], float] = {}
    for row in train_rows:
        pair = _canonical_pair(row.drug1, row.drug2)
        support_count[pair] = support_count.get(pair, 0.0) + 1.0
        neighbor_map.setdefault(pair[0], set()).add(pair[1])
        neighbor_map.setdefault(pair[1], set()).add(pair[0])

    return {"neighbor_map": neighbor_map, "support_count": support_count}


def _load_stacked_graph_context(csv_path: Path) -> Dict[str, Any]:
    rows = read_ddi_dataset(csv_path)
    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    groups = np.asarray(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in rows], dtype=object)

    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=max(2, int(round(1.0 / 0.2))), shuffle=True, random_state=42)
    dummy_x = np.zeros(len(rows), dtype=np.int8)
    train_val_idx, _ = next(splitter.split(dummy_x, labels, groups))
    train_val_rows = [rows[int(index)] for index in train_val_idx]

    labels_train_val = np.asarray([int(row.label) for row in train_val_rows], dtype=np.int64)
    groups_train_val = np.asarray(["||".join(_canonical_pair(row.drug1, row.drug2)) for row in train_val_rows], dtype=object)
    splitter2 = StratifiedGroupKFold(n_splits=max(2, int(round(1.0 / 0.2))), shuffle=True, random_state=43)
    dummy_x2 = np.zeros(len(train_val_rows), dtype=np.int8)
    train_idx, _ = next(splitter2.split(dummy_x2, labels_train_val, groups_train_val))
    train_rows = [train_val_rows[int(index)] for index in train_idx]

    neighbor_map: Dict[str, set[str]] = {}
    support_count: Dict[Tuple[str, str], float] = {}
    for row in train_rows:
        pair = _canonical_pair(row.drug1, row.drug2)
        support_count[pair] = support_count.get(pair, 0.0) + 1.0
        neighbor_map.setdefault(pair[0], set()).add(pair[1])
        neighbor_map.setdefault(pair[1], set()).add(pair[0])

    return {"neighbor_map": neighbor_map, "support_count": support_count}


class PredictionEngine:
    """Load saved models once and run deterministic inference."""

    def __init__(self) -> None:
        self._traditional_pipeline: FeaturePipeline = joblib.load(ARTIFACTS / "traditional_feature_pipeline.joblib")
        self._traditional_logreg = joblib.load(ARTIFACTS / "traditional_logreg.joblib")
        self._traditional_xgb = joblib.load(ARTIFACTS / "traditional_xgb.joblib")
        self._traditional_thresholds = self._load_thresholds(ROOT / "artifacts" / "metrics" / "traditional_ml_metrics.json")

        self._tfidf_pair_pipeline: FeaturePipeline = joblib.load(ARTIFACTS / "feature_pipeline.joblib")
        champion_artifact = joblib.load(ARTIFACTS / "champion_model.joblib")
        self._tfidf_pair_model, self._tfidf_pair_threshold = _unwrap_model_artifact(champion_artifact, fallback_threshold=0.5)

        self._graph_improved_bundle = joblib.load(ARTIFACTS / "graph_improved_feature_bundle.joblib")
        self._graph_improved_xgb = joblib.load(ARTIFACTS / "graph_improved_xgb.joblib")
        self._graph_improved_thresholds = self._load_thresholds(ROOT / "artifacts" / "metrics" / "graph_metrics_improved.json")

        gnn_checkpoint = torch.load(ARTIFACTS / "graph_gnn_champion.pt", map_location="cpu")
        self._gnn_threshold = float(gnn_checkpoint.get("threshold", 0.5))
        self._gnn_context = _build_gnn_context(DATA_PATH)
        self._gnn_model = GraphDDIGNN(
            in_dim=int(gnn_checkpoint["node_feature_dim"]),
            hidden_dim=int(gnn_checkpoint["hidden_dim"]),
            embed_dim=int(gnn_checkpoint["embed_dim"]),
            dropout=float(gnn_checkpoint["dropout"]),
        )
        self._gnn_model.load_state_dict(gnn_checkpoint["state_dict"])
        self._gnn_model.eval()

    @staticmethod
    def _load_thresholds(path: Path) -> Dict[str, float]:
        metrics = _load_json(path)
        thresholds: Dict[str, float] = {}

        models = metrics.get("models", {})
        for model_name, payload in models.items():
            threshold = payload.get("threshold")
            if threshold is not None:
                thresholds[model_name] = float(threshold)

        champion = metrics.get("champion", {})
        if isinstance(champion, dict) and champion.get("threshold") is not None:
            thresholds[str(champion.get("model_name", "champion"))] = float(champion["threshold"])

        if metrics.get("early_fusion", {}).get("metrics", {}).get("threshold") is not None:
            thresholds["early_fusion"] = float(metrics["early_fusion"]["metrics"]["threshold"])

        if metrics.get("late_fusion", {}).get("metrics", {}).get("threshold") is not None:
            thresholds["late_fusion"] = float(metrics["late_fusion"]["metrics"]["threshold"])

        return thresholds

    @staticmethod
    def _build_single_row(drug1: str, drug2: str, abstract: str) -> DDIRow:
        return DDIRow(drug1=drug1, drug2=drug2, text=abstract or "", label=0)

    @staticmethod
    def _safe_probability(model: Any, x_data: Any) -> float:
        probability = float(model.predict_proba(x_data)[:, 1][0])
        return max(0.0, min(1.0, probability))

    @staticmethod
    def _feature_vector(feature_map: Dict[str, float], feature_names: Sequence[str]) -> np.ndarray:
        return np.asarray([[float(feature_map.get(name, 0.0)) for name in feature_names]], dtype=np.float32)

    def _build_output(self, model: str, prediction: str, confidence: float | None, signal_text: str, notes: str) -> ModelOutput:
        confidence_value = 0.0 if confidence is None else float(confidence)
        interaction_type = _infer_interaction_type(signal_text, prediction)
        severity, severity_label = _infer_severity(signal_text, prediction, confidence_value)
        return ModelOutput(
            model=model,
            prediction=prediction,
            confidence=confidence,
            interaction_type=interaction_type,
            severity=severity,
            severity_label=severity_label,
            notes=notes,
        )

    @staticmethod
    def _adjusted_confidence(output: ModelOutput) -> float:
        confidence = 0.0 if output.confidence is None else float(output.confidence)
        if output.interaction_type == "General pharmacological interaction":
            confidence *= 0.7
        return max(0.0, min(1.0, confidence))

    def _traditional_logreg_result(self, drug1: str, drug2: str, abstract: str) -> ModelOutput:
        row = self._build_single_row(drug1, drug2, abstract)
        x_data, _ = transform_rows([row], self._traditional_pipeline)
        threshold = self._traditional_thresholds.get("logistic_regression", 0.5)
        probability = self._safe_probability(self._traditional_logreg, x_data)
        prediction = _evaluate_prediction(probability, threshold)
        return self._build_output(
            model="Logistic Regression",
            prediction=prediction,
            confidence=probability,
            signal_text=abstract,
            notes="TF-IDF and pair features",
        )

    def _traditional_xgb_result(self, drug1: str, drug2: str, abstract: str) -> ModelOutput:
        row = self._build_single_row(drug1, drug2, abstract)
        x_data, _ = transform_rows([row], self._traditional_pipeline)
        threshold = self._traditional_thresholds.get("xgboost", 0.5)
        probability = self._safe_probability(self._traditional_xgb, x_data)
        prediction = _evaluate_prediction(probability, threshold)
        return self._build_output(
            model="XGBoost",
            prediction=prediction,
            confidence=probability,
            signal_text=abstract,
            notes="Boosted text and pair model",
        )

    def _tfidf_pair_result(self, drug1: str, drug2: str, abstract: str) -> ModelOutput:
        row = self._build_single_row(drug1, drug2, abstract)
        x_data, _ = transform_rows([row], self._tfidf_pair_pipeline)
        probability = self._safe_probability(self._tfidf_pair_model, x_data)
        prediction = _evaluate_prediction(probability, self._tfidf_pair_threshold)
        return self._build_output(
            model="TF-IDF + Pair Champion",
            prediction=prediction,
            confidence=probability,
            signal_text=abstract,
            notes=f"Champion text feature model (threshold={self._tfidf_pair_threshold:.2f})",
        )

    def _graph_improved_result(self, drug1: str, drug2: str, abstract: str) -> ModelOutput:
        feature_map = _graph_feature_map(drug1, drug2, self._graph_improved_bundle["neighbor_map"])
        feature_names = self._graph_improved_bundle["feature_names"]
        x_data = self._feature_vector(feature_map, feature_names)
        threshold = self._graph_improved_thresholds.get("graph_improved_xgb", 0.5)
        probability = self._safe_probability(self._graph_improved_xgb, x_data)
        prediction = _evaluate_prediction(probability, threshold)
        return self._build_output(
            model="Graph-based Neo4j XGBoost",
            prediction=prediction,
            confidence=probability,
            signal_text=abstract,
            notes="Improved graph topology feature bundle",
        )

    def _gnn_result(self, drug1: str, drug2: str, abstract: str) -> ModelOutput:
        node_to_idx = self._gnn_context["node_to_idx"]
        left = drug1.strip().lower()
        right = drug2.strip().lower()
        if left not in node_to_idx or right not in node_to_idx:
            return self._build_output(
                model="Graph GNN (GraphSAGE)",
                prediction="Unavailable",
                confidence=None,
                signal_text=abstract,
                notes="At least one drug is absent from the GNN vocabulary",
            )

        pair_idx = torch.tensor([[node_to_idx[left], node_to_idx[right]]], dtype=torch.long)
        with torch.no_grad():
            logits = self._gnn_model(
                self._gnn_context["node_features"],
                self._gnn_context["edge_index"],
                pair_idx,
            )
            probability = float(torch.sigmoid(logits).cpu().numpy()[0])

        return self._build_output(
            model="Graph GNN (GraphSAGE)",
            prediction=_evaluate_prediction(probability, self._gnn_threshold),
            confidence=probability,
            signal_text=abstract,
            notes="Loaded from saved GNN checkpoint",
        )

    def _llm_structured_result(self, drug1: str, drug2: str, abstract: str) -> ModelOutput:
        text = (abstract or "").strip()
        if not text:
            return self._build_output(
                model="LLM Structured Predictor",
                prediction="No DDI",
                confidence=0.5,
                signal_text="",
                notes="Abstract omitted; deterministic neutral fallback",
            )

        prediction = predict_llm_stage4_safe(drug1=drug1, drug2=drug2, text=text, temperature=0.0)
        confidence = float(prediction.confidence_score)
        label = "DDI" if prediction.interaction else "No DDI"
        evidence = prediction.evidence_snippet or text
        return self._build_output(
            model="LLM Structured Predictor",
            prediction=label,
            confidence=confidence,
            signal_text=evidence,
            notes="Structured schema inference",
        )

    @staticmethod
    def _clinical_fusion(model_outputs: Sequence[ModelOutput], abstract: str) -> Dict[str, Any]:
        weighted_score = 0.0
        severity_weighted_sum = 0.0
        total_weight = 0.0
        ddi_votes = 0
        available_votes = 0

        llm_output = next((output for output in model_outputs if output.model == "LLM Structured Predictor"), None)

        for output in model_outputs:
            if output.prediction == "Unavailable":
                continue

            weight = MODEL_WEIGHTS.get(output.model, 0.1)
            confidence = PredictionEngine._adjusted_confidence(output)
            prediction_i = _model_binary_prediction(output)
            weighted_score += weight * confidence * prediction_i
            severity_weighted_sum += weight * _model_severity_score(output)
            total_weight += weight
            available_votes += 1
            ddi_votes += prediction_i

        if llm_output is not None and llm_output.confidence is not None and llm_output.confidence > 0.9:
            interaction_type = llm_output.interaction_type
            severity = llm_output.severity
            severity_label = llm_output.severity_label
        else:
            interaction_type = _infer_interaction_type(abstract, "DDI" if weighted_score >= 0.4 else "No DDI")
            severity, severity_label = _severity_bucket_from_score(
                1.0 if total_weight == 0 else severity_weighted_sum / total_weight
            )

        initial_prediction = _clinical_prediction_from_score(weighted_score)

        if severity_label == "Safe" and interaction_type == "No clinically meaningful interaction":
            final_prediction = "No DDI"
        elif severity_label == "Safe" and initial_prediction != "No DDI":
            final_prediction = "Mild DDI"
        elif severity in ["Moderate", "Severe"]:
            final_prediction = f"{severity} DDI"
        else:
            final_prediction = initial_prediction

        clinical_interpretation = _clinical_interpretation(final_prediction, interaction_type, severity_label)
        agreement = f"{ddi_votes}/{available_votes} models support DDI" if available_votes else "0/0 models support DDI"

        return {
            "final_prediction": final_prediction,
            "confidence": round(float(max(0.0, min(1.0, weighted_score))), 4),
            "interaction_type": interaction_type,
            "severity": severity,
            "severity_label": severity_label,
            "clinical_interpretation": clinical_interpretation,
            "model_used": "Hybrid Fusion",
            "model_agreement": agreement,
        }

    def infer_pair(self, drug1: str, drug2: str, abstract: str | None = None) -> Dict[str, Any]:
        abstract_text = (abstract or "").strip()
        evidence = _extract_evidence_sentences(drug1=drug1, drug2=drug2, abstract=abstract_text, top_k=3)

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(self._traditional_logreg_result, drug1, drug2, abstract_text),
                executor.submit(self._traditional_xgb_result, drug1, drug2, abstract_text),
                executor.submit(self._tfidf_pair_result, drug1, drug2, abstract_text),
                executor.submit(self._graph_improved_result, drug1, drug2, abstract_text),
                executor.submit(self._gnn_result, drug1, drug2, abstract_text),
                executor.submit(self._llm_structured_result, drug1, drug2, abstract_text),
            ]
            outputs = [future.result() for future in futures]

        final_summary = self._clinical_fusion(outputs, abstract_text)
        return {
            **final_summary,
            "model_outputs": [output.to_dict() for output in outputs],
            "evidence": evidence,
        }

    def score_pair(self, drug1: str, drug2: str, abstract: str | None = None) -> List[Dict[str, Any]]:
        """Backward-compatible table rows used by legacy UI path."""
        return self.infer_pair(drug1=drug1, drug2=drug2, abstract=abstract).get("model_outputs", [])


engine = PredictionEngine()