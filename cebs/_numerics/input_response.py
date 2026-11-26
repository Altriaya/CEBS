"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np

def candidate_supports(history=12, lengths=(2, 4, 6)):
    result = []
    for length in lengths:
        for start in range(1, history - length + 1):
            result.append((start, length))
    return result

def shrink_support(x, start, length, alpha):
    x = np.asarray(x)
    y = x.copy()
    stop = start + length
    if x.ndim != 3 or not 1 <= start < stop <= x.shape[1] or (not 0 < alpha <= 1):
        raise ValueError('invalid support transform')
    y[:, start:stop] = (1 - alpha) * x[:, start:stop] + alpha * x[:, start - 1:start]
    return y

def rms(x):
    return np.sqrt(np.mean(np.asarray(x, np.float64) ** 2, axis=(-2, -1)))

def response_score(f0, g0, ft, gt, std):
    base = (np.asarray(f0, np.float64) - np.asarray(g0, np.float64)) / float(std)
    transformed = (np.asarray(ft, np.float64) - np.asarray(gt, np.float64)) / float(std)
    return rms(base - transformed)

def robust_scan_calibration(scores, quantile=0.95):
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 2:
        raise ValueError('scores must be support x window')
    (q25, median, q75) = np.quantile(scores, [0.25, 0.5, 0.75], axis=1)
    scale = np.maximum(q75 - q25, 1e-12)
    normalized = (scores - median[:, None]) / scale[:, None]
    maximum = normalized.max(0)
    threshold = float(np.quantile(maximum, quantile))
    return {'median': median, 'scale': scale, 'threshold': threshold, 'maximum': maximum}

def apply_scan(scores, median, scale):
    z = (np.asarray(scores, np.float64) - np.asarray(median)[:, None]) / np.asarray(scale)[:, None]
    return (z.max(0), z.argmax(0))

def interval_iou(a, b):
    (sa, la) = a
    (sb, lb) = b
    left = max(sa, sb)
    right = min(sa + la, sb + lb)
    inter = max(0, right - left)
    union = la + lb - inter
    return inter / union

def circular_block_sample(n, block, rng):
    chunks = []
    size = min(block, n)
    while sum((len(x) for x in chunks)) < n:
        start = int(rng.integers(0, n))
        chunks.append((start + np.arange(size)) % n)
    return np.concatenate(chunks)[:n]
