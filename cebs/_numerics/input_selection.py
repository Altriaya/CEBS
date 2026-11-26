"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np
from cebs._numerics.temporal_projection import corridor_score, mask_from_score

def empirical_percentile(calibration, values):
    calibration = np.asarray(calibration, np.float64)
    values = np.asarray(values, np.float64)
    if calibration.ndim != 2 or values.ndim != 2 or calibration.shape[0] != values.shape[0]:
        raise ValueError('expected matching node-by-window matrices')
    result = np.empty_like(values)
    for node in range(len(calibration)):
        result[node] = (1.0 + (calibration[node, :, None] <= values[node, None, :]).sum(axis=0)) / (calibration.shape[1] + 1.0)
    return result

def signed_knn_evidence(percentile, neutral=0.95):
    percentile = np.asarray(percentile, np.float64)
    neutral = float(neutral)
    below = (percentile - neutral) / neutral
    above = (percentile - neutral) / (1.0 - neutral)
    return np.clip(np.where(percentile >= neutral, above, below), -1.0, 1.0)

def node_evidence(shrink_percentile, knn_percentile, reliable, shrink_quantile, knn_neutral, weight):
    candidate = np.asarray(shrink_percentile) > float(shrink_quantile)
    adjustment = float(weight) * signed_knn_evidence(knn_percentile, knn_neutral) * np.asarray(reliable, bool)
    return candidate.astype(np.float64) * np.maximum(1.0 + adjustment, 0.0)

def soft_event_mask(evidence, supports, threshold, radius, bounds):
    evidence = np.asarray(evidence, np.float64)
    supports = np.asarray(supports, np.int64)
    if evidence.ndim != 2 or len(supports) != len(evidence):
        raise ValueError('support/evidence mismatch')
    n = evidence.shape[1]
    starts = supports[:, 0]
    width = n + int(starts.max()) + 1
    score = np.zeros(width, np.float64)
    for (row, start) in enumerate(starts):
        score[int(start):int(start) + n] += evidence[row]
    ids = np.arange(width)
    candidates = np.flatnonzero((ids >= int(starts.max())) & (ids < n + int(starts.min())) & (score >= float(threshold)))
    order = candidates[np.lexsort((candidates, -score[candidates]))]
    blocked = np.zeros(width, bool)
    seeds = []
    for event in order:
        if blocked[event]:
            continue
        seeds.append(int(event))
        blocked[max(0, event - int(radius)):min(width, event + int(radius) + 1)] = True
    seeds = np.asarray(sorted(seeds), np.int64)
    return (mask_from_score(corridor_score(seeds, n, *bounds)), seeds, score)

def select_threshold(path, risk_limit):
    eligible = [float(row['threshold']) for row in path if float(row['empirical_rate']) <= float(risk_limit)]
    return min(eligible) if eligible else None
