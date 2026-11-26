"""CEBS core numerical implementation."""
from __future__ import annotations

def protocol() -> dict:
    return {'description': 'observed-target cross-event motif and input-response pairing', 'history': HISTORY, 'horizon': HORIZON, 'lengths': LENGTHS, 'features': 'first differences divided by trusted channel q75 absolute step; rounded into width .10 bins; minimum feature RMS .50', 'event_gap': EVENT_GAP, 'fit_min_independent_events': MIN_FIT_EVENTS, 'fit_min_blocks': MIN_FIT_BLOCKS, 'held_min_independent_events': MIN_QUERY_EVENTS, 'crossfit': '12 phase-aligned poisoned time blocks; block index modulo 4 assigns heldout folds; 12 trusted blocks use same fold assignment', 'null': '999 independent within-block phase-preserving shifts; max across all eligible precursor lags in each null repeat', 'pairing': 'active v2 Shrink q95 flags; count precursor votes at a shared event center; every contributing forecast history ends strictly before target motif begins; no >=3-vote gate', 'multiplicity': 'BH FDR .05 across all eligible motifs/channels/lengths in each fit fold; lag search included in permutation max', 'phase_points': PHASE, 'trusted_risk': TRUSTED_RISK}

def independent_positions(positions: np.ndarray) -> np.ndarray:
    chosen = []
    last = -EVENT_GAP
    for p in positions:
        if int(p) - last >= EVENT_GAP:
            chosen.append(int(p))
            last = int(p)
    return np.asarray(chosen, np.int64)

def percent_flags(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    out = np.zeros_like(query, bool)
    for node in range(len(reference)):
        ordered = np.sort(reference[node])
        out[node] = (1 + np.searchsorted(ordered, query[node], side='right')) / (len(ordered) + 1) > 0.95
    return out

def precursor(flags: np.ndarray, supports: np.ndarray, raw_length: int) -> np.ndarray:
    """t x lag, with no response from a history containing t or later labels."""
    value = np.zeros((raw_length, HISTORY + HORIZON), np.int8)
    for (node, support) in enumerate(supports):
        start = int(support[0])
        w = np.flatnonzero(flags[node])
        centers = w + start
        for lag in range(1, HISTORY + HORIZON + 1):
            if lag < HISTORY - start:
                continue
            t = centers + lag
            valid = t < raw_length
            value[t[valid], lag - 1] += 1
    return value

def reconstruct_trusted(trusted) -> list[np.ndarray]:
    raws = []
    for b in range(12):
        ids = np.flatnonzero(trusted['block_ids'] == b)
        if not np.array_equal(trusted['starts'][ids], np.arange(trusted['starts'][ids[0]], trusted['starts'][ids[0]] + len(ids))):
            raise AssertionError('trusted block not contiguous')
        prefix = trusted['x'][ids[0]] * float(trusted['std']) + float(trusted['mean'])
        raw = np.concatenate((prefix, trusted['y'][ids, 0], trusted['y'][ids[-1], 1:]))
        if not np.allclose(raw[np.arange(len(ids))[:, None] + HISTORY + np.arange(HORIZON)], trusted['y'][ids], atol=1e-05, rtol=1e-05):
            raise AssertionError('trusted future alignment mismatch')
        raws.append(raw)
    return raws

def motif_rows(series: np.ndarray, channel: int, length: int, scale: float):
    feature = np.lib.stride_tricks.sliding_window_view(np.diff(series[:, channel]), length - 1) / scale
    quant = np.rint(feature / BIN_WIDTH).astype(np.int32)
    nonflat = np.sqrt(np.mean(feature ** 2, axis=1)) >= MIN_FEATURE_RMS
    return (quant, nonflat)

def bh(pvalues: np.ndarray) -> np.ndarray:
    accepted = np.zeros(len(pvalues), bool)
    if not len(pvalues):
        return accepted
    order = np.argsort(pvalues)
    ok = pvalues[order] <= FDR * np.arange(1, len(pvalues) + 1) / len(pvalues)
    if ok.any():
        accepted[order[:np.flatnonzero(ok)[-1] + 1]] = True
    return accepted

def window_mask(cells: np.ndarray) -> np.ndarray:
    point = cells.any(axis=1)
    prefix = np.r_[0, np.cumsum(point)]
    w = np.arange(len(cells) - HISTORY - HORIZON + 1)
    return prefix[w + HISTORY + HORIZON] > prefix[w + HISTORY]
