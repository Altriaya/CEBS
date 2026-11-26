"""CEBS core numerical implementation."""
from __future__ import annotations

def future_scores(point):
    """Exactly equivalent to projecting selected cells onto 12-step futures."""
    per_time = np.asarray(point, np.float64).max(axis=1)
    return np.lib.stride_tricks.sliding_window_view(per_time[12:], 12).max(axis=1)

def choose_joint_threshold(base, scores, limit=0.05):
    (base, scores) = (np.asarray(base, bool), np.asarray(scores, np.float64))
    if base.shape != scores.shape or base.ndim != 1 or (not len(base)):
        raise ValueError('Expected nonempty aligned trusted windows')
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Unexpected nonfinite evidence')
    if base.mean() > limit:
        return (None, float(base.mean()))
    values = np.unique(scores[np.isfinite(scores) & (scores > 0)])
    candidates = np.unique(np.r_[np.nextafter(0.0, 1.0), values, np.nextafter(values, np.inf), np.inf])

    def risk(cut):
        return float((base | (scores >= cut)).mean())
    (lo, hi) = (0, len(candidates) - 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if risk(candidates[mid]) <= limit:
            hi = mid
        else:
            lo = mid + 1
    return (float(candidates[lo]), risk(candidates[lo]))
