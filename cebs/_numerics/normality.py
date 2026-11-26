"""CEBS core numerical implementation."""
from __future__ import annotations

def features(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.float64)
    mean = y.mean(axis=1)
    std = y.std(axis=1)
    slope = (y[:, -1] - y[:, 0]) / (y.shape[1] - 1)
    rough = np.abs(np.diff(y, axis=1)).mean(axis=1)
    return np.concatenate([mean, std, slope, rough], axis=1)

def scaled(query: np.ndarray, donors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(donors, axis=0)
    scale = np.maximum(np.quantile(donors, 0.75, axis=0) - np.quantile(donors, 0.25, axis=0), 0.0001)
    return ((query - med) / scale, (donors - med) / scale)

def knn_distance(query: np.ndarray, donors: np.ndarray, chunk: int=512) -> np.ndarray:
    out = np.empty(len(query), np.float64)
    donor_norm = np.sum(donors * donors, axis=1)
    for lo in range(0, len(query), chunk):
        q = query[lo:lo + chunk]
        d2 = np.maximum(np.sum(q * q, axis=1)[:, None] + donor_norm[None] - 2 * q @ donors.T, 0) / query.shape[1]
        nearest = np.partition(d2, K - 1, axis=1)[:, :K]
        out[lo:lo + len(q)] = np.sqrt(nearest).mean(axis=1)
    return out

def trusted_scores(raw: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
    out = np.empty(len(raw), np.float64)
    for held in range(12):
        fit = block_ids != held
        (query, donors) = scaled(raw[~fit], raw[fit])
        out[~fit] = knn_distance(query, donors)
    return out
