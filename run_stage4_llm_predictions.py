"""Stage 4: Batch LLM predictions with strict structured output.

Uses local Ollama (default model: llama3.1:8b) to predict DDI interaction
for each row and stores predictions for evaluation.

Saved artifacts:
- artifacts/predictions/stage4_llm_predictions.jsonl
- artifacts/predictions/stage4_llm_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.agents.predictor import predict_llm_stage4_safe
from src.features.pipeline import DDIRow, read_ddi_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _row_key(row: DDIRow) -> str:
    payload = "||".join([
        (row.drug1 or "").strip().lower(),
        (row.drug2 or "").strip().lower(),
        (row.text or "").strip().lower(),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def run_stage4_predictions(
    data_path: str = "data/processed/ddi_dataset.csv",
    model_name: str = "llama3.1:8b",
    max_rows: int = 0,
    output_jsonl: str = "artifacts/predictions/stage4_llm_predictions.jsonl",
    output_summary: str = "artifacts/predictions/stage4_llm_summary.json",
) -> Dict[str, Any]:
    rows = read_ddi_dataset(data_path)
    if max_rows > 0:
        rows = rows[:max_rows]

    output_path = Path(output_jsonl)
    summary_path = Path(output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_records: List[Dict[str, Any]] = []
    true_positive = 0
    true_negative = 0
    predicted_positive = 0
    predicted_negative = 0

    for index, row in enumerate(rows):
        pred = predict_llm_stage4_safe(
            drug1=row.drug1,
            drug2=row.drug2,
            text=row.text,
            model_name=model_name,
            temperature=0.0,
        )

        interaction_pred = bool(pred.interaction)
        label_true = int(row.label)

        if label_true == 1:
            true_positive += 1
        else:
            true_negative += 1

        if interaction_pred:
            predicted_positive += 1
        else:
            predicted_negative += 1

        record = {
            "row_index": index,
            "row_key": _row_key(row),
            "drug1": row.drug1,
            "drug2": row.drug2,
            "label_true": label_true,
            "interaction": interaction_pred,
            "confidence_score": float(pred.confidence_score),
            "evidence_snippet": pred.evidence_snippet,
        }
        prediction_records.append(record)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "data_path": data_path,
            "model_name": model_name,
            "max_rows": max_rows,
            "predictions_file": str(output_path),
        },
        "counts": {
            "total_rows": len(prediction_records),
            "true_positive_rows": true_positive,
            "true_negative_rows": true_negative,
            "predicted_interaction_true": predicted_positive,
            "predicted_interaction_false": predicted_negative,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Saved %d Stage 4 predictions to %s", len(prediction_records), output_path)
    logger.info("Saved Stage 4 summary to %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 strict LLM predictions")
    parser.add_argument("--data-path", default="data/processed/ddi_dataset.csv")
    parser.add_argument("--model-name", default="llama3.1:8b")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-jsonl", default="artifacts/predictions/stage4_llm_predictions.jsonl")
    parser.add_argument("--output-summary", default="artifacts/predictions/stage4_llm_summary.json")
    args = parser.parse_args()

    run_stage4_predictions(
        data_path=args.data_path,
        model_name=args.model_name,
        max_rows=args.max_rows,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
    )


if __name__ == "__main__":
    main()