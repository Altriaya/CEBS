"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np

def support_knn_reconstruct(x, donors, start, length, k=5):
    """Reconstruct one interval using trusted donor residuals and observed coordinates outside it."""
    x = np.asarray(x, np.float32)
    donors = np.asarray(donors, np.float32)
    stop = start + length
    if x.ndim != 3 or donors.ndim != 3 or x.shape[1:] != donors.shape[1:] or (not 1 <= start < stop <= x.shape[1]):
        raise ValueError('invalid histories or support')
    outside = np.r_[0:start, stop:x.shape[1]]
    q = x[:, outside].reshape(len(x), -1)
    d = donors[:, outside].reshape(len(donors), -1)
    distance = np.maximum(np.sum(q * q, axis=1)[:, None] + np.sum(d * d, axis=1)[None] - 2 * q @ d.T, 0)
    distance[distance <= 1e-10] = np.inf
    if k >= len(donors):
        raise ValueError('k must be smaller than donor count')
    idx = np.argpartition(distance, k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(distance, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.sqrt(np.take_along_axis(distance, idx, axis=1))
    finite = np.isfinite(dist)
    if not finite.all():
        raise ValueError('insufficient non-identical trusted donors')
    weight = 1 / np.maximum(dist, 1e-12)
    weight /= weight.sum(1, keepdims=True)
    residual = donors[:, start:stop] - donors[:, start - 1:start]
    average = np.sum(weight[:, :, None, None] * residual[idx], axis=1)
    transformed = x.copy()
    transformed[:, start:stop] = x[:, start - 1:start] + average
    return (transformed, dist[:, -1], idx, weight)

def amplitude_scale(calibration, raw_knn, shrink):
    calibration = np.asarray(calibration, np.float64)
    numerator = np.sqrt(np.mean((np.asarray(shrink, np.float64) - calibration) ** 2))
    denominator = np.sqrt(np.mean((np.asarray(raw_knn, np.float64) - calibration) ** 2))
    if denominator <= 0:
        raise ValueError('zero KNN intervention magnitude')
    return float(numerator / denominator)

def robust_parameters(values):
    (q25, median, q75) = np.quantile(np.asarray(values, np.float64), [0.25, 0.5, 0.75], axis=1)
    return (median, np.maximum(q75 - q25, 1e-12))

def top_candidates(z, count):
    z = np.asarray(z, np.float64)
    if not 0 < count <= z.shape[0]:
        raise ValueError('invalid candidate count')
    idx = np.argpartition(z, -count, axis=0)[-count:]
    values = np.take_along_axis(z, idx, axis=0)
    order = np.argsort(values, axis=0)[::-1]
    return np.take_along_axis(idx, order, axis=0)

def joint_margin(shrink_z, knn_z, reliable):
    result = np.minimum(np.asarray(shrink_z, np.float64), np.asarray(knn_z, np.float64))
    result[~np.asarray(reliable, bool)] = -np.inf
    return result
