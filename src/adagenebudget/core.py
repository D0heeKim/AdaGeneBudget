"""User-facing AdaGeneBudget selector."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .idf import compute_idf
from .selection import CellSelection, select_adagenebudget


class AdaGeneBudget:
    """Training-free cell-adaptive gene-token selector.

    ``fit`` estimates inverse detection-frequency weights from reference cells.
    ``select_row`` applies the paper's expression-specificity ranking and bounded
    score-mass allocation to one cell. Backbone-native ordering is intentionally
    handled by ``adagenebudget.backends`` after gene-set selection.
    """

    def __init__(self, *, tau: float = 0.90, k_min: int = 128, k_max: int = 600):
        if not (0.0 < tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if not (0 < k_min <= k_max):
            raise ValueError("require 0 < k_min <= k_max")
        self.tau = float(tau)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.idf_: np.ndarray | None = None

    def fit(self, reference_matrix) -> "AdaGeneBudget":
        self.idf_ = compute_idf(reference_matrix)
        return self

    def select_row(self, positions, values, *, tie_breaker=None) -> CellSelection:
        if self.idf_ is None:
            raise RuntimeError("fit(reference_matrix) must be called before select_row")
        return select_adagenebudget(
            values, positions, self.idf_, tau=self.tau, k_min=self.k_min,
            k_max=self.k_max, tie_breaker=tie_breaker,
        )
