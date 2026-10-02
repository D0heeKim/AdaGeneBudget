"""Reference-derived inverse detection frequency used by AdaGeneBudget."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def detection_counts(matrix) -> np.ndarray:
    """Count reference cells in which each gene is detected (expression > 0)."""
    if sparse.issparse(matrix):
        # This mirrors the validated experiment implementation and, unlike
        # merely counting stored sparse entries, does not count negative values.
        return np.asarray((matrix > 0).sum(axis=0)).reshape(-1).astype(np.int64)

    x = np.asarray(matrix)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    return np.count_nonzero(x > 0, axis=0).astype(np.int64)


def compute_idf(matrix, *, dtype=np.float32) -> np.ndarray:
    """Compute smoothed reference IDF: log((N_ref + 1)/(df_g + 1)) + 1."""
    if len(matrix.shape) != 2:
        raise ValueError("matrix must be 2-dimensional")
    n_ref = int(matrix.shape[0])
    if n_ref <= 0:
        raise ValueError("reference matrix must contain at least one cell")
    df = detection_counts(matrix).astype(np.float64, copy=False)
    idf = np.log((n_ref + 1.0) / (df + 1.0)) + 1.0
    return idf.astype(dtype, copy=False)
