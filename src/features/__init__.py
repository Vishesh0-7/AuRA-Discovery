"""Feature engineering utilities for DDI model training."""

from typing import TYPE_CHECKING

__all__ = [
	"build_feature_pipeline",
	"load_training_data",
	"transform_rows",
	"build_feature_matrix",
	"build_neighbor_map",
	"fetch_graph_rows",
	"load_graph_training_data",
	"split_rows",
]

if TYPE_CHECKING:
	from .graph_pipeline import (
		build_feature_matrix,
		build_neighbor_map,
		fetch_graph_rows,
		load_graph_training_data,
		split_rows,
	)
	from .pipeline import build_feature_pipeline, load_training_data, transform_rows


def build_feature_pipeline(*args, **kwargs):
	from .pipeline import build_feature_pipeline as _build_feature_pipeline

	return _build_feature_pipeline(*args, **kwargs)


def load_training_data(*args, **kwargs):
	from .pipeline import load_training_data as _load_training_data

	return _load_training_data(*args, **kwargs)


def transform_rows(*args, **kwargs):
	from .pipeline import transform_rows as _transform_rows

	return _transform_rows(*args, **kwargs)


def build_neighbor_map(*args, **kwargs):
	from .graph_pipeline import build_neighbor_map as _build_neighbor_map

	return _build_neighbor_map(*args, **kwargs)


def fetch_graph_rows(*args, **kwargs):
	from .graph_pipeline import fetch_graph_rows as _fetch_graph_rows

	return _fetch_graph_rows(*args, **kwargs)


def build_feature_matrix(*args, **kwargs):
	from .graph_pipeline import build_feature_matrix as _build_feature_matrix

	return _build_feature_matrix(*args, **kwargs)


def split_rows(*args, **kwargs):
	from .graph_pipeline import split_rows as _split_rows

	return _split_rows(*args, **kwargs)


def load_graph_training_data(*args, **kwargs):
	from .graph_pipeline import load_graph_training_data as _load_graph_training_data

	return _load_graph_training_data(*args, **kwargs)
