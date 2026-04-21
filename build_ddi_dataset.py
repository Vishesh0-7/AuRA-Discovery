"""
Build DDI dataset from extracted interactions in Neo4j.

Output schema:
- drug1
- drug2
- text
- label

Label rule:
- 1 if interaction validation_status == 'validated'
- 0 otherwise

Usage:
    python build_ddi_dataset.py
"""

from __future__ import annotations

import csv
import random
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
import matplotlib.pyplot as plt

from src.database.graph_connector import ResearchGraph


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_ddi_rows(graph: ResearchGraph) -> List[Dict[str, object]]:
    """Fetch DDI rows from Neo4j in dataset format."""
    query = """
    MATCH ()-[i:INTERACTS_WITH]-()
    RETURN DISTINCT
        i.drug_a AS drug1,
        i.drug_b AS drug2,
        COALESCE(i.evidence_text, '') AS text,
        CASE
            WHEN i.validation_status = 'validated' THEN 1
            ELSE 0
        END AS label
    """
    return graph.query(query)


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    left = (drug1 or "").strip().lower()
    right = (drug2 or "").strip().lower()
    return tuple(sorted((left, right)))


def _unique_drugs(rows: Sequence[Dict[str, object]]) -> List[str]:
    drugs = set()
    for row in rows:
        drug1 = str(row.get("drug1", "") or "").strip()
        drug2 = str(row.get("drug2", "") or "").strip()
        if drug1:
            drugs.add(drug1)
        if drug2:
            drugs.add(drug2)
    return sorted(drugs)


def _compute_class_stats(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    positive = sum(1 for row in rows if int(row.get("label", 0) or 0) == 1)
    negative = sum(1 for row in rows if int(row.get("label", 0) or 0) == 0)
    return {"positive": positive, "negative": negative, "total": positive + negative}


def _format_ratio(negative: int, positive: int) -> str:
    if positive == 0:
        return "n/a"
    return f"{negative / positive:.2f}:1"


def _sample_negative_pairs(
    positive_pairs: Sequence[Tuple[str, str]],
    drug_pool: Sequence[str],
    target_negative_count: int,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    if target_negative_count <= 0:
        return []

    rng = random.Random(random_state)
    positive_set = set(positive_pairs)
    sampled_set: set[Tuple[str, str]] = set()
    sampled_rows: List[Dict[str, object]] = []

    if len(drug_pool) < 2:
        raise ValueError("Need at least two unique drugs to generate negative pairs")

    max_attempts = max(10_000, target_negative_count * 50)
    attempts = 0
    while len(sampled_rows) < target_negative_count and attempts < max_attempts:
        attempts += 1
        drug1, drug2 = rng.sample(list(drug_pool), 2)
        pair = _canonical_pair(drug1, drug2)
        if pair in positive_set or pair in sampled_set:
            continue

        sampled_set.add(pair)
        sampled_rows.append(
            {
                "drug1": pair[0],
                "drug2": pair[1],
                "text": "",
                "label": 0,
            }
        )

    if len(sampled_rows) < target_negative_count:
        raise RuntimeError(
            f"Could only generate {len(sampled_rows)} negative pairs out of {target_negative_count} requested"
        )

    return sampled_rows


def _save_eda_plots(
    before_rows: Sequence[Dict[str, object]],
    balanced_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    before_stats = _compute_class_stats(before_rows)
    after_stats = _compute_class_stats(balanced_rows)

    class_fig, class_ax = plt.subplots(figsize=(8, 5))
    labels = ["Positive", "Negative"]
    before_values = [before_stats["positive"], before_stats["negative"]]
    after_values = [after_stats["positive"], after_stats["negative"]]
    x_positions = range(len(labels))
    width = 0.35

    class_ax.bar([x - width / 2 for x in x_positions], before_values, width=width, label="Before", color="#4C78A8")
    class_ax.bar([x + width / 2 for x in x_positions], after_values, width=width, label="After", color="#F58518")
    class_ax.set_xticks(list(x_positions))
    class_ax.set_xticklabels(labels)
    class_ax.set_ylabel("Sample count")
    class_ax.set_title("Class Distribution Before and After Balancing")
    class_ax.legend()
    class_ax.grid(axis="y", alpha=0.2)
    class_fig.tight_layout()
    class_fig.savefig(output_dir / "class_distribution_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(class_fig)

    def _drug_frequency(rows: Sequence[Dict[str, object]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for row in rows:
            drug1 = str(row.get("drug1", "") or "").strip()
            drug2 = str(row.get("drug2", "") or "").strip()
            if drug1:
                counts[drug1] += 1
            if drug2:
                counts[drug2] += 1
        return counts

    before_freq = _drug_frequency(before_rows).most_common(20)
    after_freq = _drug_frequency(balanced_rows).most_common(20)
    drug_labels = sorted({name for name, _ in before_freq} | {name for name, _ in after_freq})

    before_map = dict(before_freq)
    after_map = dict(after_freq)
    before_counts = [before_map.get(name, 0) for name in drug_labels]
    after_counts = [after_map.get(name, 0) for name in drug_labels]

    freq_fig, (freq_before_ax, freq_after_ax) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    freq_before_ax.bar(drug_labels, before_counts, color="#54A24B")
    freq_before_ax.set_title("Top Drug Frequency Distribution Before Balancing")
    freq_before_ax.set_ylabel("Count")
    freq_before_ax.tick_params(axis="x", rotation=60)
    freq_before_ax.grid(axis="y", alpha=0.2)

    freq_after_ax.bar(drug_labels, after_counts, color="#E45756")
    freq_after_ax.set_title("Top Drug Frequency Distribution After Balancing")
    freq_after_ax.set_ylabel("Count")
    freq_after_ax.tick_params(axis="x", rotation=60)
    freq_after_ax.grid(axis="y", alpha=0.2)

    freq_fig.tight_layout()
    freq_fig.savefig(output_dir / "drug_frequency_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(freq_fig)


def _make_balanced_rows(
    rows: Sequence[Dict[str, object]],
    negative_to_positive_ratio: float = 1.0,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    if negative_to_positive_ratio not in {1.0, 2.0}:
        raise ValueError("negative_to_positive_ratio must be 1.0 or 2.0")

    positive_rows = [row for row in rows if int(row.get("label", 0) or 0) == 1]
    positive_pairs = [_canonical_pair(str(row.get("drug1", "")), str(row.get("drug2", ""))) for row in positive_rows]
    drug_pool = _unique_drugs(rows)
    target_negative_count = int(round(len(positive_rows) * negative_to_positive_ratio))
    negative_rows = _sample_negative_pairs(
        positive_pairs=positive_pairs,
        drug_pool=drug_pool,
        target_negative_count=target_negative_count,
        random_state=random_state,
    )

    balanced_rows = list(positive_rows) + negative_rows
    rng = random.Random(random_state)
    rng.shuffle(balanced_rows)
    return balanced_rows


def save_dataset(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write dataset rows to CSV with required column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["drug1", "drug2", "text", "label"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "drug1": row.get("drug1", ""),
                    "drug2": row.get("drug2", ""),
                    "text": row.get("text", ""),
                    "label": int(row.get("label", 0)),
                }
            )


def build_ddi_dataset(
    output_file: str = "data/processed/ddi_dataset.csv",
    balanced_output_file: str = "data/processed/ddi_dataset_balanced.csv",
    negative_to_positive_ratio: float = 1.0,
) -> int:
    """Build, balance, and save the DDI dataset. Returns balanced row count."""
    load_dotenv()
    output_path = Path(output_file)
    balanced_output_path = Path(balanced_output_file)
    figures_dir = Path("reports/figures")

    logger.info("Building DDI dataset from Neo4j...")
    rows: List[Dict[str, object]] = []

    try:
        with ResearchGraph() as graph:
            rows = fetch_ddi_rows(graph)
    except Exception as exc:
        logger.warning(
            "Could not fetch interactions from Neo4j (%s). "
            "Writing an empty dataset with headers to %s.",
            exc,
            output_path,
        )

    save_dataset(rows, output_path)

    stats = _compute_class_stats(rows)
    logger.info(
        "Loaded dataset: total=%d positive=%d negative=%d ratio=%s",
        stats["total"],
        stats["positive"],
        stats["negative"],
        _format_ratio(stats["negative"], stats["positive"]),
    )

    balanced_rows = _make_balanced_rows(
        rows=rows,
        negative_to_positive_ratio=negative_to_positive_ratio,
    )
    balanced_stats = _compute_class_stats(balanced_rows)
    save_dataset(balanced_rows, balanced_output_path)
    _save_eda_plots(rows, balanced_rows, figures_dir)

    logger.info(
        "Saved raw dataset to %s and balanced dataset to %s",
        output_path,
        balanced_output_path,
    )
    logger.info(
        "Balanced dataset: total=%d positive=%d negative=%d ratio=%s",
        balanced_stats["total"],
        balanced_stats["positive"],
        balanced_stats["negative"],
        _format_ratio(balanced_stats["negative"], balanced_stats["positive"]),
    )
    logger.info("Saved EDA figures to %s", figures_dir)
    return len(balanced_rows)


if __name__ == "__main__":
    build_ddi_dataset()
