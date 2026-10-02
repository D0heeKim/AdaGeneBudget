"""Cell-adaptive score-mass budget allocation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AdaptiveBudget:
    raw_k: int
    selected_k: int
    retained_mass: float
    lower_clipped: bool
    upper_clipped: bool


def adaptive_budget(sorted_scores, *, tau: float, k_min: int, k_max: int) -> AdaptiveBudget:
    """Return the shortest prefix reaching ``tau`` score mass, then apply bounds."""
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must be in (0, 1]")
    if not (0 < k_min <= k_max):
        raise ValueError("require 0 < k_min <= k_max")

    scores = np.asarray(sorted_scores, dtype=np.float64)
    n = int(scores.size)
    if n == 0:
        return AdaptiveBudget(0, 0, 1.0, False, False)
    if np.any(scores < 0) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and nonnegative")

    total = float(scores.sum(dtype=np.float64))
    if total <= 0.0:
        raw_k = n
    else:
        cumulative = np.cumsum(scores, dtype=np.float64)
        raw_k = int(np.searchsorted(cumulative, tau * total, side="left") + 1)
        raw_k = min(raw_k, n)

    lower = bool(n >= k_min and raw_k < k_min)
    upper = bool(raw_k > k_max)
    selected_k = min(n, k_max, max(k_min, raw_k))
    retained = float(scores[:selected_k].sum(dtype=np.float64) / max(total, 1e-12))
    return AdaptiveBudget(raw_k, selected_k, retained, lower, upper)


def match_fixed_budget(nonzero_counts, target_mean: float) -> tuple[int, float]:
    """Find integer K whose realized mean min(n_i, K) best matches ``target_mean``."""
    counts = np.asarray(nonzero_counts, dtype=np.int64)
    if counts.size == 0 or np.any(counts < 0):
        raise ValueError("nonzero_counts must be a non-empty nonnegative array")
    maximum = int(counts.max())
    if maximum <= 0:
        raise ValueError("at least one cell must contain an eligible gene")

    lo, hi = 1, maximum
    best_k = 1
    best_mean = float(np.minimum(counts, 1).mean())
    best_diff = abs(best_mean - float(target_mean))

    while lo <= hi:
        k = (lo + hi) // 2
        mean = float(np.minimum(counts, k).mean())
        diff = abs(mean - float(target_mean))
        if diff < best_diff or (math.isclose(diff, best_diff) and k < best_k):
            best_k, best_mean, best_diff = k, mean, diff
        if mean < target_mean:
            lo = k + 1
        else:
            hi = k - 1

    for k in range(max(1, best_k - 3), min(maximum, best_k + 3) + 1):
        mean = float(np.minimum(counts, k).mean())
        diff = abs(mean - float(target_mean))
        if diff < best_diff or (math.isclose(diff, best_diff) and k < best_k):
            best_k, best_mean, best_diff = k, mean, diff
    return int(best_k), float(best_mean)
