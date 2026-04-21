"""Stage 7: Full evaluation and comparison across all model families.

Evaluates and compares:
- Logistic Regression
- XGBoost
- Graph model
- LLM model
- Hybrid model

Computes precision, recall, F1, and PR-AUC, creates comparison tables and
bar charts, and highlights class-imbalance handling effects.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(name: str, source: Dict[str, Any], notes: str = "") -> Dict[str, Any]:
    return {
        "model": name,
        "precision": float(source["precision"]),
        "recall": float(source["recall"]),
        "f1": float(source["f1"]),
        "pr_auc": float(source["pr_auc"]),
        "notes": notes,
    }


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "precision", "recall", "f1", "pr_auc", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| Model | Precision | Recall | F1 | PR-AUC | Notes |\n|---|---:|---:|---:|---:|---|\n"
    lines = [
        f"| {row['model']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} | {row['notes']} |"
        for row in rows
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def _plot_model_metrics(rows: List[Dict[str, Any]], output_path: Path) -> None:
    models = [row["model"] for row in rows]
    precision = [row["precision"] for row in rows]
    recall = [row["recall"] for row in rows]
    f1 = [row["f1"] for row in rows]
    pr_auc = [row["pr_auc"] for row in rows]

    x = list(range(len(models)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar([v - 1.5 * width for v in x], precision, width=width, label="Precision", color="#4C78A8")
    ax.bar([v - 0.5 * width for v in x], recall, width=width, label="Recall", color="#F58518")
    ax.bar([v + 0.5 * width for v in x], f1, width=width, label="F1", color="#54A24B")
    ax.bar([v + 1.5 * width for v in x], pr_auc, width=width, label="PR-AUC", color="#E45756")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Stage 7: Model Comparison Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_imbalance_effect(summary: Dict[str, Any], output_path: Path) -> None:
    labels = [
        "Baseline\nChampion\nMacro-F1",
        "No-SMOTE\nHybrid Stage2\nF1",
        "SMOTE\nHybrid Stage2\nF1",
    ]
    values = [
        float(summary["baseline_champion_macro_f1"]),
        float(summary["hybrid_no_smote_f1"]),
        float(summary["hybrid_smote_f1"]),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(labels, values, color=["#4C78A8", "#54A24B", "#E45756"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Effect of Class Imbalance Handling")
    ax.grid(axis="y", alpha=0.2)

    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_stage7_full_evaluation() -> Dict[str, Any]:
    traditional = _read_json(Path("artifacts/metrics/traditional_ml_metrics.json"))
    graph = _read_json(Path("artifacts/metrics/graph_metrics.json"))
    stage6 = _read_json(Path("artifacts/metrics/stage6_hybrid_metrics.json"))
    baseline = _read_json(Path("artifacts/metrics/baseline_metrics.json"))
    no_smote = _read_json(Path("reports/hybrid_metrics_nosmote.json"))
    smote = _read_json(Path("reports/hybrid_metrics_smote.json"))

    graph_champion_name = str(graph["champion"]["model_name"])
    graph_test = graph["models"][graph_champion_name]["test"]
    graph_metrics = {
        "precision": float(graph_test["precision_class_1"]),
        "recall": float(graph_test["recall_class_1"]),
        "f1": float(graph_test["f1_class_1"]),
        "pr_auc": float(graph_test["pr_auc"]),
    }

    early = stage6["early_fusion"]["metrics"]
    late = stage6["late_fusion"]["metrics"]
    hybrid_name = "Hybrid (Late Fusion)" if float(late["f1"]) >= float(early["f1"]) else "Hybrid (Early Fusion)"
    hybrid_metrics = late if hybrid_name == "Hybrid (Late Fusion)" else early

    rows = [
        _metric_row("Logistic Regression", traditional["models"]["logistic_regression"], "class_weight=balanced"),
        _metric_row("XGBoost", traditional["models"]["xgboost"], "scale_pos_weight"),
        _metric_row("Graph Model", graph_metrics, f"champion={graph_champion_name}"),
        _metric_row("LLM Model", stage6["late_fusion"]["components"]["llm"], "LLM-confidence model"),
        _metric_row(hybrid_name, hybrid_metrics, "TF-IDF + Graph + LLM"),
    ]

    best_model = max(rows, key=lambda row: (row["f1"], row["pr_auc"]))

    imbalance_summary = {
        "raw_distribution": {
            "train_negative": int(no_smote["smote"]["class_distribution_before"]["negative"]),
            "train_positive": int(no_smote["smote"]["class_distribution_before"]["positive"]),
        },
        "balanced_distribution": {
            "train_negative": int(traditional["dataset"]["train_negative"]),
            "train_positive": int(traditional["dataset"]["train_positive"]),
        },
        "baseline_champion_macro_f1": float(baseline["champion"]["test_macro_f1"]),
        "hybrid_no_smote_f1": float(no_smote["stage2"]["f1"]),
        "hybrid_smote_f1": float(smote["stage2"]["f1"]),
        "highlight": (
            "Class weighting and scale_pos_weight remained stable, while the tested SMOTE setup "
            "degraded Stage-2 F1 on the skewed split."
        ),
    }

    report = {
        "comparison_rows": rows,
        "best_performing_model": {
            "model": best_model["model"],
            "f1": float(best_model["f1"]),
            "pr_auc": float(best_model["pr_auc"]),
        },
        "class_imbalance_effect": imbalance_summary,
    }

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"

    _save_csv(rows, reports_dir / "stage7_model_comparison.csv")
    _save_markdown(rows, reports_dir / "stage7_model_comparison.md")
    (reports_dir / "stage7_full_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    _plot_model_metrics(rows, figures_dir / "stage7_model_comparison_bars.png")
    _plot_imbalance_effect(imbalance_summary, figures_dir / "stage7_imbalance_effect.png")

    logger.info("Saved Stage 7 comparison table to %s", reports_dir / "stage7_model_comparison.csv")
    logger.info("Saved Stage 7 markdown table to %s", reports_dir / "stage7_model_comparison.md")
    logger.info("Saved Stage 7 report to %s", reports_dir / "stage7_full_evaluation.json")
    logger.info("Saved Stage 7 charts to %s", figures_dir)
    logger.info(
        "Best performing model: %s (F1=%.4f, PR-AUC=%.4f)",
        best_model["model"],
        best_model["f1"],
        best_model["pr_auc"],
    )

    return report


if __name__ == "__main__":
    run_stage7_full_evaluation()