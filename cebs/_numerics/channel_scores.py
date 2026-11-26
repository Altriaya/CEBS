"""CEBS core numerical implementation."""
from __future__ import annotations

def top_mean(values: np.ndarray, fraction: float=0.05) -> float:
    flat = np.asarray(values, np.float64).reshape(-1)
    k = max(1, int(np.ceil(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - k)[-k:].mean())
