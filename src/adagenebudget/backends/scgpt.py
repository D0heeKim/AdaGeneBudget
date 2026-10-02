"""scGPT-specific post-selection ordering helpers."""
from __future__ import annotations
import numpy as np


def restore_native_order(selected_local_indices, row_positions) -> np.ndarray:
    """Restore a score-selected subset to scGPT's native/input gene order."""
    selected = np.asarray(selected_local_indices, dtype=np.int64)
    positions = np.asarray(row_positions, dtype=np.int64)
    if selected.size <= 1:
        return selected
    return selected[np.argsort(positions[selected], kind="stable")]
