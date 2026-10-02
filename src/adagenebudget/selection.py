"""Backbone-agnostic AdaGeneBudget and Fixed-TF-IDF gene-set selection."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .budget import adaptive_budget
from .scoring import (
    expression_specificity_scores,
    stable_descending_order,
    descending_order_with_tiebreak,
)


@dataclass(frozen=True)
class CellSelection:
    """Selection result in row-local coordinates.

    ``local_indices`` indexes the expressed genes supplied by the caller.  It
    represents the selected set in score-rank order; a backbone adapter can
    subsequently restore/recompute the backbone-native sequence order.
    """

    local_indices: np.ndarray
    retained_mass: float
    raw_k: int
    selected_k: int
    lower_clipped: bool = False
    upper_clipped: bool = False


def _order(scores: np.ndarray, tie_breaker=None) -> np.ndarray:
    if tie_breaker is None:
        return stable_descending_order(scores)
    return descending_order_with_tiebreak(scores, tie_breaker)


def select_adagenebudget(
    values,
    positions,
    idf,
    *,
    tau: float = 0.90,
    k_min: int = 128,
    k_max: int = 600,
    tie_breaker=None,
) -> CellSelection:
    """Select the bounded shortest prefix reaching ``tau`` TF-IDF score mass."""
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if values.shape != positions.shape:
        raise ValueError("values and positions must have the same shape")
    if values.size == 0:
        return CellSelection(np.empty(0, dtype=np.int64), 1.0, 0, 0)

    scores = expression_specificity_scores(values, idf, positions)
    order = _order(scores, tie_breaker)
    budget = adaptive_budget(scores[order], tau=tau, k_min=k_min, k_max=k_max)
    chosen = order[: budget.selected_k]
    return CellSelection(
        local_indices=chosen.astype(np.int64, copy=False),
        retained_mass=budget.retained_mass,
        raw_k=budget.raw_k,
        selected_k=budget.selected_k,
        lower_clipped=budget.lower_clipped,
        upper_clipped=budget.upper_clipped,
    )


def select_fixed_tfidf(
    values,
    positions,
    idf,
    *,
    budget: int,
    tie_breaker=None,
) -> CellSelection:
    """Fixed-budget ablation using the same expression-specificity ranking."""
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if values.shape != positions.shape:
        raise ValueError("values and positions must have the same shape")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if values.size == 0:
        return CellSelection(np.empty(0, dtype=np.int64), 1.0, 0, 0)

    scores = expression_specificity_scores(values, idf, positions)
    order = _order(scores, tie_breaker)
    k = min(int(values.size), int(budget))
    chosen = order[:k]
    total = float(scores.sum(dtype=np.float64))
    retained = float(scores[chosen].sum(dtype=np.float64) / max(total, 1e-12))
    return CellSelection(chosen.astype(np.int64, copy=False), retained, k, k)
