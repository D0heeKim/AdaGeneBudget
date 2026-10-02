"""Geneformer-specific post-selection native rank-value ordering."""
from __future__ import annotations
import numpy as np
from ..scoring import descending_order_with_tiebreak


def native_scores(raw_values, gene_positions, gene_medians, *, n_counts: float | None = None) -> np.ndarray:
    values = np.asarray(raw_values, dtype=np.float64)
    positions = np.asarray(gene_positions, dtype=np.int64)
    medians = np.asarray(gene_medians, dtype=np.float64)
    scores = values / medians[positions]
    # Kang's validated implementation includes (10,000 / n_counts), which is
    # constant within a cell and therefore leaves the ordering unchanged.
    if n_counts is not None:
        if not np.isfinite(n_counts) or n_counts <= 0:
            raise ValueError("n_counts must be finite and positive")
        scores = scores * (10_000.0 / float(n_counts))
    if not np.isfinite(scores).all():
        raise ValueError("non-finite Geneformer native score")
    return scores


def reorder_subset_by_native_rank(selected_local_indices, row_positions, row_values,
                                  gene_medians, *, token_ids=None, n_counts=None) -> np.ndarray:
    selected = np.asarray(selected_local_indices, dtype=np.int64)
    positions = np.asarray(row_positions, dtype=np.int64)
    values = np.asarray(row_values, dtype=np.float64)
    if selected.size == 0:
        return selected
    pos = positions[selected]
    vals = values[selected]
    scores = native_scores(vals, pos, gene_medians, n_counts=n_counts)
    tie = pos if token_ids is None else np.asarray(token_ids, dtype=np.int64)[pos]
    order = descending_order_with_tiebreak(scores, tie)
    return selected[order]
