"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np

def robust_location_scale(values, axis=-1):
    values = np.asarray(values, np.float64)
    (q25, median, q75) = np.quantile(values, (0.25, 0.5, 0.75), axis=axis)
    return (median, np.maximum(q75 - q25, 1e-08))

def horizon_excess_fit(predicted_poison, predicted_reference, target, std, top_fraction):
    """Positive excess fit of the infected model, separately for every horizon."""
    f = np.asarray(predicted_poison, np.float64)
    g = np.asarray(predicted_reference, np.float64)
    y = np.asarray(target, np.float64)
    if f.shape != g.shape or f.shape != y.shape or f.ndim != 3:
        raise ValueError('predictions and target must have shape [window,horizon,channel]')
    excess = ((g - y) / float(std)) ** 2 - ((f - y) / float(std)) ** 2
    excess = np.maximum(excess, 0.0)
    channels = excess.shape[-1]
    k = max(1, int(np.ceil(channels * float(top_fraction))))
    return np.partition(excess, channels - k, axis=-1)[..., -k:].mean(axis=-1).T

def binding_cube(input_score, horizon_score):
    a = np.asarray(input_score, np.float64)
    b = np.asarray(horizon_score, np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('scores must be [support,window] and [horizon,window]')
    return np.minimum(a[:, None, :], b[None, :, :])

def template_strength(joint, threshold, clip):
    joint = np.asarray(joint, np.float64)
    finite = np.where(np.isfinite(joint), joint, -np.inf)
    excess = np.clip(finite - float(threshold), 0.0, float(clip))
    return excess.mean(axis=-1)

def conditional_permutation(starts, rng, period=288):
    """Permute days within time-of-day so the null preserves periodic marginals."""
    starts = np.asarray(starts, np.int64)
    order = np.arange(len(starts), dtype=np.int64)
    for phase in np.unique(starts % int(period)):
        ids = np.flatnonzero(starts % int(period) == phase)
        order[ids] = rng.permutation(ids)
    return order

def nms_templates(score, supports, count, support_iou_threshold, horizon_radius):
    score = np.asarray(score, np.float64)
    supports = np.asarray(supports, np.int64)
    flat = np.argsort(-score.ravel(), kind='mergesort')
    chosen = []
    for value in flat:
        (support_index, horizon) = np.unravel_index(int(value), score.shape)
        (start, length) = supports[support_index]
        reject = False
        for (prior_support, prior_horizon) in chosen:
            (pstart, plength) = supports[prior_support]
            overlap = max(0, min(start + length, pstart + plength) - max(start, pstart))
            iou = overlap / (length + plength - overlap)
            if iou >= support_iou_threshold and abs(horizon - prior_horizon) <= horizon_radius:
                reject = True
                break
        if not reject:
            chosen.append((int(support_index), int(horizon)))
            if len(chosen) == int(count):
                break
    return chosen

def exact_budget_alarm(score, count):
    score = np.asarray(score, np.float64)
    finite = np.flatnonzero(np.isfinite(score))
    if len(finite) < count:
        raise ValueError('insufficient finite scores')
    order = np.lexsort((finite, -score[finite]))
    alarm = np.zeros(len(score), bool)
    alarm[finite[order[:count]]] = True
    return alarm

def jaccard(a, b):
    (a, b) = (np.asarray(a, bool), np.asarray(b, bool))
    union = a | b
    return float((a & b).sum() / union.sum()) if union.any() else 1.0
