"""Graph feature engineering for DDI link prediction.

Builds edge-level topology features from Neo4j interaction data:
- common neighbors
- degree
- Jaccard similarity

The graph features are computed using a train-only adjacency map to
reduce leakage when evaluating on held-out edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.database.graph_connector import ResearchGraph


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphDDIRow:
    """One unique drug pair extracted from Neo4j."""

    drug1: str
    drug2: str
    label: int
    support_count: int
    validated_count: int


def _canonical_pair(drug_a: str, drug_b: str) -> Tuple[str, str]:
    left = (drug_a or "").strip().lower()
    right = (drug_b or "").strip().lower()
    return tuple(sorted((left, right)))


def fetch_graph_rows(graph: ResearchGraph) -> List[GraphDDIRow]:
    """Fetch unique drug pairs from Neo4j interactions."""
    query = """
    MATCH ()-[i:INTERACTS_WITH]-()
    RETURN
        i.drug_a AS drug_a,
        i.drug_b AS drug_b,
        CASE WHEN i.validation_status = 'validated' THEN 1 ELSE 0 END AS label,
        COALESCE(i.validation_status, 'unknown') AS validation_status
    """

    records = graph.query(query)
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for record in records:
        drug1, drug2 = _canonical_pair(record.get("drug_a", ""), record.get("drug_b", ""))
        pair_key = (drug1, drug2)
        label = int(record.get("label", 0) or 0)

        if pair_key not in aggregated:
            aggregated[pair_key] = {
                "drug1": drug1,
                "drug2": drug2,
                "label": label,
                "support_count": 1,
                "validated_count": 1 if label == 1 else 0,
            }
        else:
            aggregated[pair_key]["support_count"] += 1
            aggregated[pair_key]["validated_count"] += 1 if label == 1 else 0
            aggregated[pair_key]["label"] = max(aggregated[pair_key]["label"], label)

    rows = [GraphDDIRow(**value) for value in aggregated.values()]
    rows.sort(key=lambda row: (row.drug1, row.drug2))
    return rows


def split_rows(
    rows: Sequence[GraphDDIRow],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[GraphDDIRow], List[GraphDDIRow], List[GraphDDIRow]]:
    """Split graph rows into train, validation, and test sets."""
    from sklearn.model_selection import train_test_split

    if not rows:
        raise ValueError("No graph rows available for splitting")

    labels = np.array([row.label for row in rows], dtype=np.int64)
    train_val_rows, test_rows = train_test_split(
        list(rows),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_val_labels = np.array([row.label for row in train_val_rows], dtype=np.int64)
    train_rows, val_rows = train_test_split(
        train_val_rows,
        test_size=val_size,
        random_state=random_state + 1,
        stratify=train_val_labels,
    )

    return list(train_rows), list(val_rows), list(test_rows)


def build_neighbor_map(rows: Sequence[GraphDDIRow]) -> Dict[str, set[str]]:
    """Build an undirected adjacency map from rows."""
    neighbors: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not row.drug1 or not row.drug2:
            continue
        neighbors[row.drug1].add(row.drug2)
        neighbors[row.drug2].add(row.drug1)
    return dict(neighbors)


def build_component_map(neighbor_map: Dict[str, set[str]]) -> Dict[str, int]:
    """Build connected-component ids for fast path-existence checks."""
    component_map: Dict[str, int] = {}
    component_id = 0

    for node in neighbor_map:
        if node in component_map:
            continue

        stack = [node]
        component_map[node] = component_id
        while stack:
            current = stack.pop()
            for neighbor in neighbor_map.get(current, set()):
                if neighbor not in component_map:
                    component_map[neighbor] = component_id
                    stack.append(neighbor)

        component_id += 1

    return component_map


def compute_pagerank(
    neighbor_map: Dict[str, set[str]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Compute PageRank scores using power iteration on an undirected graph."""
    nodes = list(neighbor_map.keys())
    node_count = len(nodes)
    if node_count == 0:
        return {}

    scores = {node: 1.0 / node_count for node in nodes}
    teleport = (1.0 - damping) / node_count

    for _ in range(max_iter):
        new_scores = {node: teleport for node in nodes}
        dangling_mass = 0.0

        for node in nodes:
            neighbors = neighbor_map.get(node, set())
            degree = len(neighbors)
            if degree == 0:
                dangling_mass += scores[node]
                continue

            contribution = damping * (scores[node] / degree)
            for neighbor in neighbors:
                if neighbor in new_scores:
                    new_scores[neighbor] += contribution

        if dangling_mass > 0.0:
            dangling_contribution = damping * dangling_mass / node_count
            for node in nodes:
                new_scores[node] += dangling_contribution

        delta = sum(abs(new_scores[node] - scores[node]) for node in nodes)
        scores = new_scores
        if delta < tol:
            break

    return scores


def compute_node2vec_embeddings(
    neighbor_map: Dict[str, set[str]],
    embedding_dim: int = 16,
) -> Optional[Dict[str, np.ndarray]]:
    """Try Node2Vec embeddings; fall back to spectral embeddings when unavailable."""

    def _spectral_fallback() -> Optional[Dict[str, np.ndarray]]:
        nodes = sorted(neighbor_map.keys())
        if not nodes:
            return None

        node_index = {node: idx for idx, node in enumerate(nodes)}
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

        for node, neighbors in neighbor_map.items():
            i = node_index[node]
            for neighbor in neighbors:
                j = node_index.get(neighbor)
                if j is None:
                    continue
                adjacency[i, j] = 1.0

        if adjacency.sum() == 0.0:
            return None

        dims = max(2, min(embedding_dim, adjacency.shape[0] - 1, adjacency.shape[1] - 1))
        if dims < 2:
            return None

        svd = TruncatedSVD(n_components=dims, random_state=42)
        embedded = svd.fit_transform(adjacency)
        embeddings = {node: embedded[idx].astype(np.float32) for node, idx in node_index.items()}
        logger.info("Using spectral embedding fallback for %d graph nodes", len(embeddings))
        return embeddings

    try:
        import networkx as nx  # type: ignore
        from node2vec import Node2Vec  # type: ignore
    except Exception:
        logger.info("Node2Vec dependencies not available; using spectral embedding fallback")
        return _spectral_fallback()

    if not neighbor_map:
        return None

    graph = nx.Graph()
    for node, neighbors in neighbor_map.items():
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)

    if graph.number_of_nodes() == 0:
        return None

    try:
        node2vec = Node2Vec(
            graph,
            dimensions=embedding_dim,
            walk_length=20,
            num_walks=50,
            workers=1,
            quiet=True,
        )
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
    except Exception as exc:
        logger.warning("Node2Vec training failed; using spectral embedding fallback: %s", exc)
        return _spectral_fallback()

    embeddings: Dict[str, np.ndarray] = {}
    for node in graph.nodes():
        try:
            embeddings[str(node)] = np.asarray(model.wv[str(node)], dtype=np.float32)
        except Exception:
            continue

    if not embeddings:
        return None
    logger.info("Computed Node2Vec embeddings for %d graph nodes", len(embeddings))
    return embeddings


def _shortest_path_length_without_direct_edge(
    source: str,
    target: str,
    neighbor_map: Dict[str, set[str]],
    max_depth: int = 6,
) -> float:
    """BFS shortest path length while excluding the direct source-target edge."""
    if source == target:
        return 0.0
    if source not in neighbor_map or target not in neighbor_map:
        return float(max_depth + 1)

    visited = {source}
    frontier = deque([(source, 0)])

    while frontier:
        current, depth = frontier.popleft()
        if depth >= max_depth:
            continue

        neighbors = set(neighbor_map.get(current, set()))
        # Exclude the direct edge between source and target to reduce leakage.
        if current == source and target in neighbors:
            neighbors.remove(target)

        for neighbor in neighbors:
            if neighbor == target:
                return float(depth + 1)
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))

    return float(max_depth + 1)


def _topology_features(
    drug1: str,
    drug2: str,
    neighbor_map: Dict[str, set[str]],
    component_map: Dict[str, int],
    pagerank_map: Dict[str, float],
    node2vec_embeddings: Optional[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    neighbors_1 = set(neighbor_map.get(drug1, set())) - {drug2}
    neighbors_2 = set(neighbor_map.get(drug2, set())) - {drug1}
    node_count = max(1.0, float(len(neighbor_map)))

    degree_1 = float(len(neighbors_1))
    degree_2 = float(len(neighbors_2))
    degree_centrality_1 = degree_1 / max(1.0, node_count - 1.0)
    degree_centrality_2 = degree_2 / max(1.0, node_count - 1.0)
    common_neighbors = neighbors_1 & neighbors_2
    common_count = float(len(common_neighbors))
    union_count = float(len(neighbors_1 | neighbors_2))
    jaccard = common_count / union_count if union_count else 0.0
    path_existence = float(
        drug1 in component_map
        and drug2 in component_map
        and component_map.get(drug1) == component_map.get(drug2)
    )
    shortest_path_len = _shortest_path_length_without_direct_edge(drug1, drug2, neighbor_map)
    shortest_path_inv = 1.0 / (1.0 + shortest_path_len)
    pagerank_1 = float(pagerank_map.get(drug1, 0.0))
    pagerank_2 = float(pagerank_map.get(drug2, 0.0))

    node2vec_cosine = 0.0
    node2vec_l2 = 0.0
    if node2vec_embeddings is not None:
        v1 = node2vec_embeddings.get(drug1)
        v2 = node2vec_embeddings.get(drug2)
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
        "shortest_path_length": shortest_path_len,
        "shortest_path_inverse": shortest_path_inv,
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
        "support_count": 0.0,
        "support_count_log": 0.0,
    }


def build_feature_matrix(
    rows: Sequence[GraphDDIRow],
    neighbor_map: Dict[str, set[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert graph rows to an X matrix and y vector."""
    if not rows:
        raise ValueError("No rows provided for feature matrix construction")

    feature_names = [
        "degree_1",
        "degree_2",
        "degree_centrality_1",
        "degree_centrality_2",
        "degree_centrality_sum",
        "degree_sum",
        "degree_log_sum",
        "degree_diff",
        "common_neighbors",
        "shared_neighbors",
        "common_neighbors_ratio",
        "jaccard",
        "path_existence",
        "shortest_path_length",
        "shortest_path_inverse",
        "pagerank_1",
        "pagerank_2",
        "pagerank_sum",
        "pagerank_diff",
        "node2vec_cosine",
        "node2vec_l2",
        "adamic_adar",
        "resource_allocation",
        "salton",
        "sorensen",
        "hub_promoted",
        "hub_depressed",
        "union_size",
        "preferential_attachment",
        "support_count",
        "support_count_log",
    ]

    feature_rows: List[List[float]] = []
    labels: List[int] = []
    component_map = build_component_map(neighbor_map)
    pagerank_map = compute_pagerank(neighbor_map)
    node2vec_embeddings = compute_node2vec_embeddings(neighbor_map)

    for row in rows:
        features = _topology_features(
            row.drug1,
            row.drug2,
            neighbor_map,
            component_map,
            pagerank_map,
            node2vec_embeddings,
        )
        features["support_count"] = float(row.support_count)
        features["support_count_log"] = math.log1p(float(row.support_count))

        feature_rows.append([float(features[name]) for name in feature_names])
        labels.append(int(row.label))

    x = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, feature_names


def load_graph_training_data(
    graph: ResearchGraph,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fetch graph rows, split them, and return graph-based training data."""
    rows = fetch_graph_rows(graph)
    train_rows, val_rows, test_rows = split_rows(
        rows,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    train_neighbor_map = build_neighbor_map(train_rows)

    x_train, y_train, feature_names = build_feature_matrix(train_rows, train_neighbor_map)
    x_val, y_val, _ = build_feature_matrix(val_rows, train_neighbor_map)
    x_test, y_test, _ = build_feature_matrix(test_rows, train_neighbor_map)

    return {
        "rows": rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "neighbor_map": train_neighbor_map,
        "feature_names": feature_names,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }
