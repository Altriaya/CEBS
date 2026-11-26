"""CEBS core numerical implementation."""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from cebs.rng import load_torch,restore_rng,rng_state,save_torch,seed_all
from cebs.io import read_json,sha,write_json
from cebs.checkpoints import inspect_best,recover_best
ARCHITECTURES=("FEDformer","TimesNet","SimpleTM")

def _model_class(architecture: str):
    """Import only the requested victim so config listing needs no ML extras."""
    if architecture == 'FEDformer':
        from models.predictors.FEDformer import Model
    elif architecture == 'TimesNet':
        from models.predictors.TimesNet import Model
    elif architecture == 'SimpleTM':
        try:
            from model.SimpleTM import Model
        except ModuleNotFoundError as exc:
            if exc.name not in ('model', 'model.SimpleTM'):
                raise
            raise ModuleNotFoundError(
                'SimpleTM requires the upstream checkout on the Python module search path. '
                'See Optional SimpleTM predictor in README.md.'
            ) from exc
    else:
        raise KeyError(f'unknown victim {architecture!r}; choose from {list(ARCHITECTURES)}')
    return Model

def namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: namespace(item) for (key, item) in value.items()})
    return value

def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def model_hash(model: 'ForecastModel') -> str:
    digest = hashlib.sha256()
    for (name, value) in sorted(model.net.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def runtime_environment() -> dict:
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'einops', 'PyWavelets', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), platform=platform.system(), packages=packages, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())

class ForecastModel(torch.nn.Module):

    def __init__(self, architecture: str, config: dict, mean: float, std: float, device: str):
        super().__init__()
        self.architecture = architecture
        self.net = _model_class(architecture)(namespace(config))
        self.mean = float(mean)
        self.std = float(std)
        self.horizon = int(config['pred_len'])
        self.to(device)

    def forward(self, x: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.mean) / self.std
        decoder = normalized.new_zeros(len(x), self.horizon, x.shape[2])
        output = self.net(normalized, marks, decoder, None)
        if isinstance(output, tuple):
            output = output[0]
        return output * self.std + self.mean

class MaskedWindows:

    def __init__(self, series: np.ndarray, marks: np.ndarray, hard: np.ndarray, local: np.ndarray, device: str, history: int=12, horizon: int=12):
        if local.shape != (len(hard), horizon, series.shape[1]):
            raise ValueError('local mask shape mismatch')
        positions = np.flatnonzero(~hard)
        if np.any(local[positions].all(axis=(1, 2))):
            raise ValueError('a retained window has every supervision cell masked')
        self.series = torch.as_tensor(series, dtype=torch.float32, device=device)
        self.marks = torch.as_tensor(marks, dtype=torch.float32, device=device)
        self.positions = torch.as_tensor(positions, dtype=torch.long, device=device)
        self.keep_cells = torch.as_tensor(~local, dtype=torch.bool, device=device)
        self.history = history
        self.horizon = horizon
        self.device = device

    def __len__(self) -> int:
        return len(self.positions)

    def batch(self, indexes: torch.Tensor):
        starts = self.positions[indexes]
        x_offset = torch.arange(self.history, device=self.device)
        y_offset = torch.arange(self.horizon, device=self.device)
        return (self.series[starts[:, None] + x_offset], self.series[starts[:, None] + self.history + y_offset], self.marks[starts[:, None] + x_offset], self.keep_cells[starts])

def _validation_mae(model: ForecastModel, data: MaskedWindows, batch_size: int, device: str) -> float:
    model.eval()
    (total, cells) = (0.0, 0)
    with torch.inference_mode():
        order = torch.arange(len(data), device=device)
        for lo in range(0, len(data), batch_size):
            (x, y, marks, _) = data.batch(order[lo:lo + batch_size])
            total += float(torch.abs(model(x, marks) - y).sum())
            cells += y.numel()
    return total / cells

def _initial_model(architecture: str, model_cfg: dict, spec: dict, device: str) -> ForecastModel:
    seed_all(1)
    model = ForecastModel(architecture, model_cfg, spec['mean'], spec['std'], device)
    if architecture == 'FEDformer' and spec.get('fedformer_initial_state'):
        initial = load_torch(spec['fedformer_initial_state'])
        model.net.load_state_dict(initial['model'])
    return model

def _train(out: Path, architecture: str, model_cfg: dict, spec: dict, protocol: dict, hard: np.ndarray, local: np.ndarray, full_series: np.ndarray, full_marks: np.ndarray, device: str) -> Path:
    folder = out / 'cebs'
    folder.mkdir(parents=True, exist_ok=True)
    (final, resume) = (folder / 'best.pth', folder / 'resume.pth')
    provenance_path = folder / 'provenance.json'
    source_hash = sha(spec['poison'])
    compatibility = {'source_hash': source_hash, 'hard_mask_hash': protocol['hard_mask_sha256'], 'local_mask_hash': protocol['local_mask_sha256'], 'initial_hash': protocol['initial_hash'], 'config_sha256': protocol['config_sha256'], 'implementation_sha256': protocol['implementation_sha256'], 'runtime_environment': protocol.get('runtime_environment', runtime_environment())}
    epochs = int(protocol['epochs'])
    if provenance_path.exists():
        old = read_json(provenance_path)
        if old.get('completed_epochs') != epochs or any((old.get(k) != v for (k, v) in compatibility.items())):
            raise RuntimeError('completed checkpoint belongs to another frozen protocol')
        if not final.exists():
            raise RuntimeError('Completed best checkpoint is missing; recovery files preserved')
        inspect_best(final.read_bytes(), old.get('checkpoint_sha256'), compatibility, architecture, model_cfg, epochs, old.get('epoch'), old.get('best_validation_mae'))
        return final
    (state, best_blob) = (None, None)
    if resume.exists():
        state = load_torch(resume)
        if any((state.get(k) != v for (k, v) in compatibility.items())):
            raise RuntimeError('resume checkpoint belongs to another frozen protocol')
        if not isinstance(state.get('epoch'), int) or not 1 <= state['epoch'] <= epochs:
            raise RuntimeError('Invalid resume epoch; files preserved')
        if not isinstance(state.get('best_epoch'), int) or not 1 <= state['best_epoch'] <= state['epoch']:
            raise RuntimeError('Invalid resume best epoch; files preserved')
        (best_blob, _) = recover_best(final, state, compatibility, architecture, model_cfg, epochs)
    elif final.exists():
        raise RuntimeError('Orphan best checkpoint without resume/provenance; files preserved')
    with np.load(spec['poison'], allow_pickle=False) as source:
        series = source['series'].astype(np.float32)
        marks = source['marks'].astype(np.float32) if 'marks' in source.files else np.zeros((len(series), 4), np.float32)
    if len(series) - int(protocol['history']) - int(protocol['horizon']) + 1 != len(hard):
        raise RuntimeError('poison training geometry changed')
    data = MaskedWindows(series, marks, hard, local, device, protocol['history'], protocol['horizon'])
    val_series = full_series[int(spec['train_end']):int(spec['validation_end'])]
    val_marks = full_marks[int(spec['train_end']):int(spec['validation_end'])]
    val_hard = np.zeros(len(val_series) - protocol['history'] - protocol['horizon'] + 1, bool)
    val_local = np.zeros((len(val_hard), protocol['horizon'], int(spec['channels'])), bool)
    validation = MaskedWindows(val_series, val_marks, val_hard, val_local, device, protocol['history'], protocol['horizon'])
    model = _initial_model(architecture, model_cfg, spec, device)
    if model_hash(model) != protocol['initial_hash']:
        raise RuntimeError('Seed-1 victim initialization changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol['learning_rate']))
    (start, best_epoch, best_mae, elapsed) = (0, None, float('inf'), 0.0)
    if state is not None:
        model.net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        start = int(state['epoch'])
        best_epoch = state['best_epoch']
        best_mae = float(state['best_validation_mae'])
        elapsed = float(state['elapsed_seconds'])
        del state
    (epochs, batch_size) = (int(protocol['epochs']), int(protocol['batch_size']))
    for epoch in range(start, epochs):
        tick = time.monotonic()
        model.train()
        order = torch.randperm(len(data), device=device)
        (loss_sum, cell_count) = (0.0, 0)
        for lo in range(0, len(data), batch_size):
            (x, y, marks, keep) = data.batch(order[lo:lo + batch_size])
            optimizer.zero_grad(set_to_none=True)
            per_cell = F.smooth_l1_loss(model(x, marks), y, reduction='none')
            denominator = keep.sum()
            if int(denominator) == 0:
                raise RuntimeError('empty masked-loss batch')
            loss = (per_cell * keep).sum() / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite CEBS training loss')
            loss.backward()
            optimizer.step()
            loss_sum += float((per_cell.detach() * keep).sum())
            cell_count += int(denominator)
        score = _validation_mae(model, validation, batch_size, device)
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite validation MAE; previous recovery files preserved')
        elapsed += time.monotonic() - tick
        if score < best_mae:
            (best_mae, best_epoch) = (score, epoch + 1)
            save_torch(final, {'model': {k: v.detach().cpu().clone() for (k, v) in model.net.state_dict().items()}, 'epoch': best_epoch, 'best_validation_mae': best_mae, 'role': 'cebs', 'architecture': architecture, 'retained_windows': len(data), 'deleted_windows': int(hard.sum()), 'locally_masked_cells': int(local[~hard].sum()), 'model_config': model_cfg, **compatibility})
            best_blob = final.read_bytes()
        save_torch(resume, {'model': model.net.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_validation_mae': best_mae, 'elapsed_seconds': elapsed, **compatibility, 'best_checkpoint_bytes': best_blob, 'best_checkpoint_sha256': hashlib.sha256(best_blob).hexdigest()})
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps({'case': spec['case'], 'victim': architecture, 'epoch': epoch + 1, 'train_loss': loss_sum / cell_count, 'validation_mae': score, 'best_epoch': best_epoch, 'best_validation_mae': best_mae, 'elapsed_seconds': elapsed}), flush=True)
    expected_best_hash = hashlib.sha256(best_blob).hexdigest()
    best = inspect_best(final.read_bytes(), expected_best_hash, compatibility, architecture, model_cfg, epochs, best_epoch, best_mae)
    summary = {key: value for (key, value) in best.items() if key != 'model'}
    summary.update({'completed_epochs': epochs, 'elapsed_seconds': elapsed, 'checkpoint_sha256': sha(final)})
    write_json(provenance_path, summary)
    resume.unlink(missing_ok=True)
    del model, optimizer, data, validation
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return final

def _evaluate(checkpoint: Path, architecture: str, model_cfg: dict, spec: dict, batch_size: int, device: str) -> dict:
    with np.load(spec['evaluation'], allow_pickle=False) as source:
        ev = {key: source[key].copy() for key in source.files}
    clean_x = ev['clean_x'].astype(np.float32)
    trigger_x = ev['trigger_x'].astype(np.float32)
    truth = ev['truth'].astype(np.float32)
    marks = ev['x_marks'].astype(np.float32) if 'x_marks' in ev else np.zeros((len(truth), 12, 4), np.float32)
    channels = ev['channels'].astype(int)
    pattern_len = int(ev['pattern_len'])
    model = ForecastModel(architecture, model_cfg, spec['mean'], spec['std'], device)
    model.net.load_state_dict(load_torch(checkpoint)['model'])
    model.eval()
    (clean_rows, trigger_rows) = ([], [])
    with torch.inference_mode():
        for lo in range(0, len(truth), batch_size):
            mark = torch.as_tensor(marks[lo:lo + batch_size], device=device)
            clean_rows.append(model(torch.as_tensor(clean_x[lo:lo + batch_size], device=device), mark).cpu().numpy())
            trigger_rows.append(model(torch.as_tensor(trigger_x[lo:lo + batch_size], device=device), mark).cpu().numpy())
    clean_pred = np.concatenate(clean_rows)
    trigger_pred = np.concatenate(trigger_rows)
    target_pred = trigger_pred[:, :pattern_len, :][:, :, channels]
    attack_target = ev['attack_target']
    if attack_target.shape != target_pred.shape:
        attack_target = attack_target[:, :pattern_len, :][:, :, channels]
    truth_target = truth[:, :pattern_len, :][:, :, channels]
    return {'normal_mae': float(np.abs(clean_pred - truth).mean()), 'trigger_truth_mae': float(np.abs(target_pred - truth_target).mean()), 'attack_target_mae': float(np.abs(target_pred - attack_target).mean())}
