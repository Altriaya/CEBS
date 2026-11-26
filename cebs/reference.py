"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

def block_split(blocks, held, inner=None):
    blocks = np.asarray(blocks)
    labels = np.unique(blocks)
    if blocks.ndim != 1 or held not in labels or len(labels) < 3:
        raise ValueError('Invalid outer block partition')
    if inner is not None and (inner == held or inner not in labels):
        raise ValueError('Invalid inner block')
    excluded = [held] if inner is None else [held, inner]
    fit = ~np.isin(blocks, excluded)
    query = blocks == (held if inner is None else inner)
    if not fit.any() or not query.any() or np.any(fit & query):
        raise ValueError('Empty or overlapping split')
    return (fit, query)

def require_physical_disjoint(starts, blocks, held, history=12, horizon=12):
    """Exclude shared raw points, not merely different window-row IDs."""
    (fit, query) = block_split(blocks, held)
    offsets = np.arange(history + horizon)
    a = np.unique((np.asarray(starts)[fit, None] + offsets).ravel())
    b = np.unique((np.asarray(starts)[query, None] + offsets).ravel())
    overlap = np.intersect1d(a, b)
    if len(overlap):
        raise ValueError(f'Held block overlaps fitting raw points: {len(overlap)}')
    return dict(fit_windows=int(fit.sum()), held_windows=int(query.sum()), fit_raw_points=int(len(a)), held_raw_points=int(len(b)), shared_raw_points=0)

def fit_readout(features, base, y_raw, keep, mean, std, ridge=10.0):
    """Same standardized ridge readout as the frozen historical A0 fit.

    Slice *before* estimating means/scales. Changing excluded labels cannot
    change the fitted parameters. This is a trusted linear reference refit,
    not an infected-backbone or victim training run.
    """
    keep = np.asarray(keep, bool)
    features = np.asarray(features, np.float64)
    base = np.asarray(base, np.float64)
    y_raw = np.asarray(y_raw, np.float64)
    if keep.shape != (len(features),) or not keep.any() or std <= 0 or (ridge <= 0):
        raise ValueError('Invalid ridge fitting population')
    x = features[keep].reshape(-1, features.shape[-1])
    y = ((y_raw[keep] - float(mean)) / float(std) - base[keep]).reshape(-1, y_raw.shape[-1])
    (xm, scale, ym) = (x.mean(0), np.maximum(x.std(0), 1e-08), y.mean(0))
    z = (x - xm) / scale
    gram = z.T @ z / len(z)
    gram.flat[::len(gram) + 1] += ridge
    cross = z.T @ (y - ym) / len(z)
    coef = cho_solve(cho_factor(gram, lower=True, check_finite=False), cross, check_finite=False)
    weights = coef / scale[:, None]
    return (weights.astype(np.float32), (ym - xm @ weights).astype(np.float32))

def nested_normality(raw_features, blocks, held, scale_fn, distance_fn):
    """Outer held uses 11 donors; each fitting block uses 10 donors.

    Scaling is independently refitted on each donor population. The caller
    supplies the unchanged historical IQR scaling/distance implementations.
    """
    result = np.empty(len(blocks), np.float64)
    records = []
    for query_block in np.unique(blocks):
        inner = None if query_block == held else query_block
        (fit, test) = block_split(blocks, held, inner)
        (query, donors) = scale_fn(raw_features[test], raw_features[fit])
        result[test] = distance_fn(query, donors)
        records.append(dict(query_block=int(query_block), donor_blocks=np.unique(blocks[fit]).astype(int).tolist(), donor_rows=int(fit.sum())))
    return (result, records)
REQUIRED_FULL_PIPELINE_STAGES = ('input_a0_and_knn', 'path_a_discovery_and_calibration', 'recurrence_scale_pairing_and_stable_core', 'channel_and_local_masks', 'normal_reference_and_base_routing', 'event_parameters_and_cut_selection', 'held_hard_and_local_risk')

def require_complete_pipeline(stages):
    missing = [s for s in REQUIRED_FULL_PIPELINE_STAGES if stages.get(s) != 'complete']
    if missing:
        raise RuntimeError('Full pipeline risk is unavailable; incomplete stages: ' + ', '.join(missing))
