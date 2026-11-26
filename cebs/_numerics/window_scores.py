"""CEBS core numerical implementation."""
from __future__ import annotations

def top_mean_rows(flat: np.ndarray) -> np.ndarray:
    k = max(1, int(np.ceil(flat.shape[1] * TOP_FRACTION)))
    return np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
