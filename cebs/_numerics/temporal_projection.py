"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np

def response_offsets(active_nodes, history):
    """Map aligned input-support nodes to downstream absolute response offsets."""
    values = []
    for node in active_nodes:
        support_start = int(node['support'][0])
        horizon_zero_based = int(node['horizon_one_based']) - 1
        values.append(int(history) + horizon_zero_based - support_start)
    if not values:
        raise ValueError('at least one active trajectory node is required')
    return np.asarray(values, np.int64)

def corridor_bounds(offsets, history, horizon, padding=0):
    """Window-start interval whose supervision contains a discovered response."""
    offsets = np.asarray(offsets, np.int64)
    if offsets.size == 0:
        raise ValueError('response offsets cannot be empty')
    lower = int(offsets.min()) - (int(history) + int(horizon) - 1) - int(padding)
    upper = int(offsets.max()) - int(history) + int(padding)
    return (lower, upper)

def corridor_score(seeds, n, lower, upper):
    """Count event-response corridors covering each window, with a tiny center tie-break."""
    score = np.zeros(int(n), np.float64)
    half_width = max((int(upper) - int(lower)) / 2.0, 0.5)
    for seed in np.asarray(seeds, np.int64):
        lo = max(0, int(seed) + int(lower))
        hi = min(int(n), int(seed) + int(upper) + 1)
        if lo >= hi:
            continue
        center = int(seed) + (int(lower) + int(upper)) / 2.0
        tie = 1.0 - np.abs(np.arange(lo, hi) - center) / (half_width + 1.0)
        score[lo:hi] += 1.0 + 0.001 * tie
    return score

def mask_from_score(score, baseline=None):
    mask = np.asarray(score) > 0
    if baseline is not None:
        mask = mask | np.asarray(baseline, bool)
    return mask
