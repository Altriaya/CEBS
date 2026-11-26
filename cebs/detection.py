"""CEBS core numerical implementation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
PERIODS = {'PEMS03': 288, 'ETTm1': 96, 'Weather': 144}

def read_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}

def validate_inputs(observed: dict, trusted: dict) -> dict:
    """Check geometry/resources, without querying a model or reading role labels."""
    required = {'x', 'y', 'starts', 'block_ids', 'mean', 'std'}
    if not required.issubset(trusted) or 'series' not in observed:
        raise ValueError('Expected series and trusted x/y/starts/block_ids/mean/std arrays')
    series = observed['series']
    if series.ndim != 2 or len(series) < 24 or (not np.isfinite(series).all()):
        raise ValueError('series must be finite physical-time x channel records, length >= 24')
    (rows, channels) = (len(trusted['starts']), series.shape[1])
    if trusted['x'].shape != (rows, 12, channels) or trusted['y'].shape != (rows, 12, channels):
        raise ValueError('Trusted histories/futures must have shape (rows,12,channels)')
    (starts, blocks) = (trusted['starts'], trusted['block_ids'])
    if starts.ndim != 1 or blocks.shape != (rows,) or starts.dtype.kind not in 'iu' or (blocks.dtype.kind not in 'iu'):
        raise ValueError('Trusted starts/block_ids must be matching integer vectors')
    if not np.array_equal(np.unique(blocks), np.arange(12)) or len(np.unique(starts)) != rows:
        raise ValueError('Twelve distinct blocks (IDs 0..11) and unique starts are required')
    if starts.min() < len(series):
        raise ValueError('Released protocol obtains trusted records from the validation interval, not training')
    footprints = []
    for block in range(12):
        selected = np.sort(starts[blocks == block])
        if not np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)):
            raise ValueError('Windows within each trusted block must be consecutive')
        points = np.unique((selected[:, None] + np.arange(24)).ravel())
        if any((np.intersect1d(points, previous).size for previous in footprints)):
            raise ValueError('Physical trusted blocks must be disjoint')
        footprints.append(points)
    raw_points = sum(map(len, footprints))
    if raw_points / len(series) > 0.05:
        raise ValueError('Trusted physical-point budget exceeds 5% of training length')
    (mean, std) = (trusted['mean'], trusted['std'])
    if np.size(mean) != 1 or np.size(std) != 1 or (not np.isfinite(float(mean))) or (not np.isfinite(float(std))) or (float(std) <= 0):
        raise ValueError('mean/std must be finite shared global scalars with std > 0')
    if not np.isfinite(trusted['x']).all() or not np.isfinite(trusted['y']).all():
        raise ValueError('Trusted arrays must be finite')
    for (record, shape) in ((observed, (len(series), 4)), (trusted, (rows, 12, 4))):
        if 'marks' in record and (record['marks'].shape != shape or not np.isfinite(record['marks']).all()):
            raise ValueError('marks must use four time features and matching temporal geometry')
    return dict(training_points=len(series), windows=len(series) - 23, channels=channels, trusted_windows=rows, trusted_blocks=12, trusted_raw_points=raw_points, trusted_fraction_of_training_length=raw_points / len(series), history=12, horizon=12, role_labels_read=False, model_queried=False)

def detect(observed_path: Path, trusted_path: Path, checkpoint: Path, output: Path, dataset: str, *, period: int | None=None, batch: int=64, channel_checkpoint: Path | None=None) -> dict:
    (observed, trusted) = (read_arrays(observed_path), read_arrays(trusted_path))
    validate_inputs(observed, trusted)
    if batch <= 0:
        raise ValueError('batch must be positive')
    if output.exists():
        raise FileExistsError('Use a new output directory; no old masks/caches are substituted')
    for path in (checkpoint, channel_checkpoint):
        if path is not None and (not Path(path).is_file()):
            raise FileNotFoundError(path)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Fresh detection uses CUDA; --preflight-only and CPU tests need no GPU')
    from cebs._detector.runner import run_detector
    return run_detector(observed_path.parent, output, dataset.lower() + '_custom', dataset, trusted, observed, checkpoint, period or PERIODS[dataset], batch, channel_checkpoint)
