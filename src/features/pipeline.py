"""Stage 3 feature pipeline for DDI classification.

Builds a sparse feature matrix using:
1. TF-IDF features from the evidence text.
2. Drug-pair encoding features from drug1/drug2 fields.
3. Normalized numeric pair features.

The main entry point is `load_training_data`, which returns X and y
ready for model training.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class FeaturePipeline:
    """Container for feature transformers and output metadata."""

    tfidf_vectorizer: TfidfVectorizer
    pair_id_vectorizer: DictVectorizer
    numeric_scaler: StandardScaler
    feature_names: List[str]


@dataclass
class MLReadySplit:
    """Train/test matrices and labels ready for classical ML models."""

    x_train: csr_matrix
    x_test: csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: FeaturePipeline


@dataclass
class DDIRow:
    """Represents one row from the DDI training CSV."""

    drug1: str
    drug2: str
    text: str
    label: int


def _tokenize_name(name: str) -> List[str]:
    return TOKEN_PATTERN.findall((name or "").lower())


def _canonical_pair(drug1: str, drug2: str) -> Tuple[str, str]:
    a = (drug1 or "").strip().lower()
    b = (drug2 or "").strip().lower()
    return tuple(sorted((a, b)))


def _pair_features(drug1: str, drug2: str) -> Dict[str, float | str]:
    d1 = (drug1 or "").strip().lower()
    d2 = (drug2 or "").strip().lower()
    t1 = set(_tokenize_name(d1))
    t2 = set(_tokenize_name(d2))

    overlap = 0.0
    if t1 or t2:
        overlap = len(t1 & t2) / max(1, len(t1 | t2))

    pair_a, pair_b = _canonical_pair(d1, d2)

    # Keep this intentionally simple and interpretable for baseline experiments.
    return {
        "pair_id": f"{pair_a}__{pair_b}",
        "same_drug": float(d1 == d2 and d1 != ""),
        "name_len_diff": float(abs(len(d1) - len(d2))),
        "token_overlap": float(overlap),
        "same_start_char": float(bool(d1 and d2 and d1[0] == d2[0])),
    }


def _pair_numeric_features(drug1: str, drug2: str) -> Dict[str, float]:
    d1 = (drug1 or "").strip().lower()
    d2 = (drug2 or "").strip().lower()
    t1 = set(_tokenize_name(d1))
    t2 = set(_tokenize_name(d2))

    union = t1 | t2
    shared = t1 & t2
    length_1 = len(d1)
    length_2 = len(d2)

    return {
        "same_drug": float(d1 == d2 and d1 != ""),
        "name_len_diff": float(abs(length_1 - length_2)),
        "token_overlap": float(len(shared) / max(1, len(union))),
        "same_start_char": float(bool(d1 and d2 and d1[0] == d2[0])),
        "shared_token_count": float(len(shared)),
        "length_ratio": float(min(length_1, length_2) / max(1, max(length_1, length_2))),
    }


def read_ddi_dataset(csv_path: str | Path) -> List[DDIRow]:
    """Read CSV rows with schema: drug1, drug2, text, label."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: List[DDIRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            label_str = str(raw.get("label", "0")).strip() or "0"
            try:
                label = int(label_str)
            except ValueError:
                label = 0

            rows.append(
                DDIRow(
                    drug1=str(raw.get("drug1", "") or ""),
                    drug2=str(raw.get("drug2", "") or ""),
                    text=str(raw.get("text", "") or ""),
                    label=label,
                )
            )

    return rows


def build_feature_pipeline(
    rows: Sequence[DDIRow],
    max_tfidf_features: int = 20000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[csr_matrix, np.ndarray, FeaturePipeline]:
    """Create a combined sparse matrix X and label vector y."""
    if not rows:
        raise ValueError("No rows provided for feature building")

    texts = [row.text for row in rows]
    labels = np.array([int(row.label) for row in rows], dtype=np.int64)

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_tfidf_features,
        min_df=min_df,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    x_text = tfidf.fit_transform(texts)

    pair_id_dicts = [{"pair_id": _pair_features(row.drug1, row.drug2)["pair_id"]} for row in rows]
    pair_id_vec = DictVectorizer(sparse=True)
    x_pair_id = pair_id_vec.fit_transform(pair_id_dicts)

    numeric_rows = [_pair_numeric_features(row.drug1, row.drug2) for row in rows]
    numeric_feature_names = list(numeric_rows[0].keys()) if numeric_rows else []
    numeric_matrix = np.asarray([[float(values[name]) for name in numeric_feature_names] for values in numeric_rows], dtype=np.float32)
    numeric_scaler = StandardScaler()
    x_numeric = numeric_scaler.fit_transform(numeric_matrix)

    x = hstack([x_text, x_pair_id, csr_matrix(x_numeric)], format="csr")

    text_feature_names = [f"tfidf::{name}" for name in tfidf.get_feature_names_out()]
    pair_feature_names = [f"pair::{name}" for name in pair_id_vec.get_feature_names_out()]
    numeric_feature_names = [f"pair_num::{name}" for name in numeric_feature_names]
    pipeline = FeaturePipeline(
        tfidf_vectorizer=tfidf,
        pair_id_vectorizer=pair_id_vec,
        numeric_scaler=numeric_scaler,
        feature_names=text_feature_names + pair_feature_names + numeric_feature_names,
    )

    return x, labels, pipeline


def transform_rows(
    rows: Sequence[DDIRow],
    pipeline: FeaturePipeline,
) -> Tuple[csr_matrix, np.ndarray]:
    """Transform rows using an already-fitted feature pipeline."""
    if not rows:
        raise ValueError("No rows provided for transformation")

    texts = [row.text for row in rows]
    labels = np.array([int(row.label) for row in rows], dtype=np.int64)

    x_text = pipeline.tfidf_vectorizer.transform(texts)
    pair_id_dicts = [{"pair_id": _pair_features(row.drug1, row.drug2)["pair_id"]} for row in rows]
    x_pair_id = pipeline.pair_id_vectorizer.transform(pair_id_dicts)

    numeric_rows = [_pair_numeric_features(row.drug1, row.drug2) for row in rows]
    numeric_feature_names = [name for name in numeric_rows[0].keys()] if numeric_rows else []
    numeric_matrix = np.asarray([[float(values[name]) for name in numeric_feature_names] for values in numeric_rows], dtype=np.float32)
    x_numeric = pipeline.numeric_scaler.transform(numeric_matrix)

    x = hstack([x_text, x_pair_id, csr_matrix(x_numeric)], format="csr")
    return x, labels


def load_training_data(
    csv_path: str | Path = "data/processed/ddi_dataset.csv",
    max_tfidf_features: int = 20000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[csr_matrix, np.ndarray, FeaturePipeline]:
    """Load dataset and return X, y (plus fitted pipeline metadata)."""
    rows = read_ddi_dataset(csv_path)
    return build_feature_pipeline(
        rows=rows,
        max_tfidf_features=max_tfidf_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )


def load_ml_ready_split(
    csv_path: str | Path = "data/processed/ddi_dataset_balanced.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    max_tfidf_features: int = 5000,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> MLReadySplit:
    """Load dataset, split into train/test, and return ML-ready matrices."""
    rows = read_ddi_dataset(csv_path)
    if not rows:
        raise ValueError("No rows available in the dataset")

    train_rows, test_rows = train_test_split(
        list(rows),
        test_size=test_size,
        random_state=random_state,
        stratify=[row.label for row in rows],
    )

    x_train, y_train, pipeline = build_feature_pipeline(
        rows=train_rows,
        max_tfidf_features=max_tfidf_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )
    x_test, y_test = transform_rows(test_rows, pipeline)

    return MLReadySplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        pipeline=pipeline,
    )


if __name__ == "__main__":
    split = load_ml_ready_split()
    print(f"X_train shape: {split.x_train.shape}")
    print(f"X_test shape: {split.x_test.shape}")
    print(f"y_train shape: {split.y_train.shape}")
    print(f"y_test shape: {split.y_test.shape}")
    print(f"Train positives: {int((split.y_train == 1).sum())}")
    print(f"Train negatives: {int((split.y_train == 0).sum())}")
    print(f"Test positives: {int((split.y_test == 1).sum())}")
    print(f"Test negatives: {int((split.y_test == 0).sum())}")
    print(f"Total features: {len(split.pipeline.feature_names)}")
