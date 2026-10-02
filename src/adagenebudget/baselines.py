"""Simple selection baselines used alongside AdaGeneBudget."""
from __future__ import annotations

import numpy as np

from .selection import CellSelection
from .scoring import stable_descending_order, descending_order_with_tiebreak


def select_top_expression(values, *, budget: int, tie_breaker=None) -> CellSelection:
    """Keep the highest-expression genes at a fixed budget.

    ``tie_breaker`` can be supplied to reproduce a backbone-specific
    deterministic secondary key (for example Geneformer token ID).
    """
    values = np.asarray(values, dtype=np.float64)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if values.size == 0:
        return CellSelection(np.empty(0, dtype=np.int64), 1.0, 0, 0)

    if tie_breaker is None:
        order = stable_descending_order(values)
    else:
        order = descending_order_with_tiebreak(values, tie_breaker)

    k = min(int(values.size), int(budget))
    chosen = order[:k]
    total = float(values.sum(dtype=np.float64))
    retained = float(values[chosen].sum(dtype=np.float64) / max(total, 1e-12))
    return CellSelection(chosen.astype(np.int64, copy=False), retained, k, k)


def select_random(n_genes: int, *, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample expressed genes without replacement.

    RNG construction is deliberately left to the experiment driver because the
    validated scGPT and Geneformer baselines used different stream-management
    conventions.
    """
    if n_genes < 0 or budget <= 0:
        raise ValueError("n_genes must be nonnegative and budget positive")
    k = min(int(n_genes), int(budget))
    if k == n_genes:
        return np.arange(n_genes, dtype=np.int64)
    return np.asarray(rng.choice(n_genes, size=k, replace=False), dtype=np.int64)
