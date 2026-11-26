"""CEBS core numerical implementation."""
from __future__ import annotations

def mapped_recurrence(raw: np.ndarray, n_windows: int) -> np.ndarray:
    return raw[np.arange(n_windows)[:, None] + HISTORY + np.arange(HORIZON)]

def mapped_trusted_recurrence(raw: np.ndarray, block_ids: np.ndarray, channels: int) -> np.ndarray:
    result = np.full((len(block_ids), HORIZON, channels), -np.inf, np.float64)
    for block in range(BLOCKS):
        ids = np.flatnonzero(block_ids == block)
        if raw.shape[1] != len(ids) + HISTORY + HORIZON - 1:
            raise AssertionError((raw.shape, len(ids)))
        for (local, row) in enumerate(ids):
            result[row] = raw[block, local + HISTORY + np.arange(HORIZON)]
    return result
