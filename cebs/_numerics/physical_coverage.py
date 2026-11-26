"""CEBS core numerical implementation."""
from __future__ import annotations

def physical_coverage(hard: np.ndarray, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (n, horizon, channels) = local.shape
    total = np.zeros((n + horizon - 1, channels), np.int16)
    removed = np.zeros_like(total)
    for h in range(horizon):
        t = np.arange(n) + h
        total[t] += 1
        removed[t] += (hard[:, None] | local[:, h]).astype(np.int16)
    return (removed, total)
