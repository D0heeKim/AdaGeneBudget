"""Expression-specificity scoring and deterministic ranking utilities."""
from __future__ import annotations

import numpy as np


def expression_specificity_scores(values, idf, positions=None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(idf, dtype=np.float64)
    if positions is not None:
        weights = weights[np.asarray(positions, dtype=np.int64)]
    if values.shape != weights.shape:
        raise ValueError("values and selected IDF weights must have the same shape")
    scores = values * weights
    if np.any(scores < 0) or not np.isfinite(scores).all():
        raise ValueError("expression-specificity scores must be finite and nonnegative")
    return scores


def stable_descending_order(scores) -> np.ndarray:
    """Descending scores; ties preserve the incoming order (scGPT semantics)."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.argsort(-scores, kind="stable").astype(np.int64, copy=False)


def descending_order_with_tiebreak(scores, tie_breaker) -> np.ndarray:
    """Descending scores with an explicit ascending deterministic tie-break."""
    scores = np.asarray(scores, dtype=np.float64)
    tie = np.asarray(tie_breaker)
    if scores.shape != tie.shape:
        raise ValueError("scores and tie_breaker must have the same shape")
    return np.lexsort((tie, -scores)).astype(np.int64, copy=False)
