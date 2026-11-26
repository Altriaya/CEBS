"""CEBS core numerical implementation."""
from __future__ import annotations

def robust_parameters(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    return (location, max(scale, 1e-12))

def evidence_logit(values: np.ndarray, trusted_size: int) -> np.ndarray:
    epsilon = 1.0 / (2.0 * (trusted_size + 1.0))
    clipped = np.clip(np.asarray(values, np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)

def pooled_lag_one_correlation(standardized: np.ndarray, block_ids: np.ndarray, starts: np.ndarray, fit: np.ndarray) -> float:
    left: list[float] = []
    right: list[float] = []
    for block in range(BLOCKS):
        ids = np.flatnonzero((block_ids == block) & fit)
        if len(ids) < 2:
            continue
        ids = ids[np.argsort(starts[ids], kind='mergesort')]
        consecutive = np.diff(starts[ids]) == 1
        left.extend(standardized[ids[:-1][consecutive]].tolist())
        right.extend(standardized[ids[1:][consecutive]].tolist())
    if len(left) < 3:
        return 0.0
    rho = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    if not np.isfinite(rho):
        return 0.0
    return float(np.clip(rho, 0.0, 0.95))

def normalized_event_scores(proposal: np.ndarray, standardized: np.ndarray, block_ids: np.ndarray, starts: np.ndarray, rho_by_block: dict[int, float]) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    output = np.full(len(proposal), -np.inf, np.float64)
    events: list[dict[str, float | int]] = []
    for block in range(BLOCKS):
        ids = np.flatnonzero((block_ids == block) & proposal)
        if not len(ids):
            continue
        ids = ids[np.argsort(starts[ids], kind='mergesort')]
        event = [int(ids[0])]
        for index in ids[1:]:
            previous = event[-1]
            if int(starts[index]) == int(starts[previous]) + 1:
                event.append(int(index))
            else:
                finish_event(event, block, standardized, starts, rho_by_block[block], output, events)
                event = [int(index)]
        finish_event(event, block, standardized, starts, rho_by_block[block], output, events)
    return (output, events)

def finish_event(indices: list[int], block: int, standardized: np.ndarray, starts: np.ndarray, rho: float, output: np.ndarray, events: list[dict[str, float | int]]) -> None:
    length = len(indices)
    variance = float(length)
    if length > 1 and rho > 0:
        lags = np.arange(1, length, dtype=np.float64)
        variance += float(2.0 * np.sum((length - lags) * np.power(rho, lags)))
    statistic = float(standardized[indices].sum() / np.sqrt(max(variance, 1e-12)))
    output[indices] = statistic
    events.append({'block': int(block), 'start': int(starts[indices[0]]), 'end': int(starts[indices[-1]]), 'windows': int(length), 'rho': float(rho), 'statistic': statistic})
