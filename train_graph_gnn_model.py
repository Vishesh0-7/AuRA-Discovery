"""Train a GraphSAGE-style GNN for DDI edge classification.

This script trains a true message-passing graph neural network over the
drug interaction graph using train-only edges to reduce leakage.

Outputs:
- artifacts/models/graph_gnn_champion.pt
- artifacts/models/graph_gnn_bundle.joblib
- artifacts/metrics/graph_gnn_metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src.database.graph_connector import ResearchGraph
from src.features.graph_pipeline import GraphDDIRow, build_neighbor_map, compute_pagerank, fetch_graph_rows, split_rows
from src.features.pipeline import read_ddi_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    best: Dict[str, float] | None = None
    best_key: Tuple[float, float, float] = (-1.0, -1.0, -1.0)

    for threshold in np.linspace(0.10, 0.90, 33):
        metrics = _evaluate(y_true, y_prob, float(threshold))
        candidate_key = (metrics["f1"], metrics["recall"], metrics["pr_auc"])
        if best is None or candidate_key > best_key:
            best = metrics
            best_key = candidate_key

    if best is None:
        raise RuntimeError("Failed to find best threshold")
    return best


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    a = (drug1 or "").strip().lower()
    b = (drug2 or "").strip().lower()
    return tuple(sorted((a, b)))


def _load_rows_from_csv(csv_path: Path) -> List[GraphDDIRow]:
    ddi_rows = read_ddi_dataset(csv_path)
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in ddi_rows:
        left, right = _canonical_pair(row.drug1, row.drug2)
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

    loaded_rows = [GraphDDIRow(**value) for value in aggregated.values()]
    loaded_rows.sort(key=lambda row: (row.drug1, row.drug2))
    return loaded_rows


def _load_rows_with_fallback(csv_path: Path) -> Tuple[List[GraphDDIRow], str]:
    try:
        with ResearchGraph() as graph:
            rows = fetch_graph_rows(graph)
        logger.info("Using Neo4j as graph source")
        return rows, "neo4j"
    except Exception as exc:
        logger.warning("Neo4j unavailable (%s). Falling back to %s", exc, csv_path)
        rows = _load_rows_from_csv(csv_path)
        logger.info("Using CSV fallback as graph source")
        return rows, "csv_fallback"


def _build_node_features(
    train_rows: Sequence[GraphDDIRow],
    all_rows: Sequence[GraphDDIRow],
) -> Tuple[torch.Tensor, Dict[str, int], torch.Tensor]:
    train_neighbor_map = build_neighbor_map(train_rows)
    pagerank_map = compute_pagerank(train_neighbor_map)

    node_set: set[str] = set()
    for row in all_rows:
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

    if not edge_pairs:
        raise ValueError("No train edges available for GNN message passing")

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return torch.tensor(features, dtype=torch.float32), node_to_idx, edge_index


def _rows_to_pairs(rows: Sequence[GraphDDIRow], node_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    pairs: List[List[int]] = []
    labels: List[float] = []
    for row in rows:
        if row.drug1 not in node_to_idx or row.drug2 not in node_to_idx:
            continue
        pairs.append([node_to_idx[row.drug1], node_to_idx[row.drug2]])
        labels.append(float(row.label))

    if not pairs:
        raise ValueError("No edge-label rows available for training/evaluation")

    pair_idx = torch.tensor(pairs, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)
    return pair_idx, y


def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    num_nodes = x.size(0)
    feat_dim = x.size(1)

    agg = x.new_zeros((num_nodes, feat_dim))
    agg.index_add_(0, dst, x[src])

    deg = x.new_zeros((num_nodes,))
    deg.index_add_(0, dst, x.new_ones((dst.size(0),)))
    return agg / deg.clamp_min(1.0).unsqueeze(1)


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neigh = _mean_aggregate(x, edge_index)
        out = self.self_linear(x) + self.neigh_linear(neigh)
        return F.relu(out)


class GraphDDIGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GraphSAGELayer(in_dim, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.conv1(x, edge_index)
        z = self.dropout(z)
        z = self.conv2(z, edge_index)
        z = self.dropout(z)
        return z

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        u = z[pair_idx[:, 0]]
        v = z[pair_idx[:, 1]]
        edge_features = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        logits = self.edge_mlp(edge_features).squeeze(1)
        return logits


def train_graph_gnn_model(
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    dropout: float = 0.25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 250,
    patience: int = 35,
    data_path: str = "data/processed/ddi_dataset.csv",
) -> Dict[str, Any]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    rows, data_source = _load_rows_with_fallback(Path(data_path))

    train_rows, val_rows, test_rows = split_rows(
        rows,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    node_features, node_to_idx, edge_index = _build_node_features(train_rows, rows)
    train_pair_idx, y_train = _rows_to_pairs(train_rows, node_to_idx)
    val_pair_idx, y_val = _rows_to_pairs(val_rows, node_to_idx)
    test_pair_idx, y_test = _rows_to_pairs(test_rows, node_to_idx)

    device = torch.device("cpu")
    model = GraphDDIGNN(
        in_dim=node_features.size(1),
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    train_pair_idx = train_pair_idx.to(device)
    val_pair_idx = val_pair_idx.to(device)
    test_pair_idx = test_pair_idx.to(device)
    y_train = y_train.to(device)

    positive = max(1.0, float(y_train.sum().item()))
    negative = max(1.0, float(y_train.numel() - y_train.sum().item()))
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state: Dict[str, torch.Tensor] | None = None
    best_val: Dict[str, float] | None = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(node_features, edge_index, train_pair_idx)
        loss = criterion(train_logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(node_features, edge_index, val_pair_idx)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()

        y_val_np = y_val.cpu().numpy()
        val_metrics = _find_best_threshold(y_val_np, val_prob)
        improved = best_val is None or (val_metrics["f1"], val_metrics["pr_auc"]) > (
            best_val["f1"],
            best_val["pr_auc"],
        )

        if improved:
            best_val = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            logger.info(
                "epoch=%d train_loss=%.4f val_f1=%.4f val_pr_auc=%.4f threshold=%.2f",
                epoch,
                float(loss.item()),
                val_metrics["f1"],
                val_metrics["pr_auc"],
                val_metrics["threshold"],
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if best_state is None or best_val is None:
        raise RuntimeError("GNN training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(node_features, edge_index, test_pair_idx)
        test_prob = torch.sigmoid(test_logits).cpu().numpy()

    threshold = float(best_val["threshold"])
    y_test_np = y_test.cpu().numpy()
    test_metrics = _evaluate(y_test_np, test_prob, threshold=threshold)

    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "graph_gnn_champion.pt"
    torch.save(
        {
            "state_dict": best_state,
            "node_feature_dim": int(node_features.shape[1]),
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "dropout": dropout,
            "threshold": threshold,
            "node_to_idx": node_to_idx,
        },
        model_path,
    )
    joblib.dump(
        {
            "node_to_idx": node_to_idx,
            "feature_names": ["degree_centrality", "pagerank", "support_log"],
        },
        models_dir / "graph_gnn_bundle.joblib",
    )

    report = {
        "config": {
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "patience": patience,
            "data_source": data_source,
            "data_path": data_path,
        },
        "dataset": {
            "total_rows": len(rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_class_0": int(sum(1 for row in train_rows if row.label == 0)),
            "train_class_1": int(sum(1 for row in train_rows if row.label == 1)),
            "test_class_0": int(sum(1 for row in test_rows if row.label == 0)),
            "test_class_1": int(sum(1 for row in test_rows if row.label == 1)),
            "node_count": len(node_to_idx),
            "message_passing_edges": int(edge_index.size(1)),
        },
        "model": {
            "name": "graphsage_edge_mlp",
            "best_epoch": best_epoch,
            "validation": best_val,
            "test": test_metrics,
        },
    }

    metrics_path = metrics_dir / "graph_gnn_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Saved GNN model to %s", model_path)
    logger.info("Saved GNN metrics to %s", metrics_path)
    logger.info(
        "GNN test precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["pr_auc"],
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphSAGE GNN for DDI prediction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    args = parser.parse_args()

    train_graph_gnn_model(
        random_state=args.random_state,
        test_size=args.test_size,
        val_size=args.val_size,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()