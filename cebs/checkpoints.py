"""CEBS core numerical implementation."""
from __future__ import annotations

import hashlib
import io
import math
import os
from pathlib import Path
import torch

def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

def inspect_best(blob, expected_hash, compatibility, architecture, model_config, epochs, best_epoch=None, best_mae=None):
    if not expected_hash or digest(blob) != expected_hash:
        raise RuntimeError('Best checkpoint file hash mismatch')
    try:
        state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    except Exception as exc:
        raise RuntimeError('Best checkpoint is unreadable; recovery files preserved') from exc
    if any((state.get(k) != v for (k, v) in compatibility.items())):
        raise RuntimeError('Best checkpoint belongs to another frozen protocol')
    if state.get('role') != 'cebs' or state.get('architecture') != architecture or state.get('model_config') != model_config:
        raise RuntimeError('Best checkpoint model identity mismatch')
    (epoch, score) = (state.get('epoch'), state.get('best_validation_mae'))
    if not isinstance(epoch, int) or not 1 <= epoch <= epochs or (not isinstance(score, (int, float))) or (not math.isfinite(score)):
        raise RuntimeError('Invalid best checkpoint epoch/validation score')
    if best_epoch is not None and (epoch != best_epoch or score != best_mae):
        raise RuntimeError('Best checkpoint and training progress disagree')
    if not isinstance(state.get('model'), dict) or not state['model']:
        raise RuntimeError('Best checkpoint has no model weights')
    if any((not torch.isfinite(v).all() for v in state['model'].values() if torch.is_tensor(v))):
        raise RuntimeError('Nonfinite best checkpoint weights')
    return state

def recover_best(path: Path, resume: dict, compatibility, architecture, model_config, epochs):
    """The atomically saved resume contains its own exact best-file bytes.

    A crash after publishing a newer best but before publishing resume must roll
    back to resume's best before repeating the interrupted epoch.
    """
    blob = resume.get('best_checkpoint_bytes')
    if not isinstance(blob, bytes):
        raise RuntimeError('Resume lacks a verified best snapshot; files preserved')
    state = inspect_best(blob, resume.get('best_checkpoint_sha256'), compatibility, architecture, model_config, epochs, resume.get('best_epoch'), resume.get('best_validation_mae'))
    if not path.exists() or digest(path.read_bytes()) != digest(blob):
        temporary = path.with_suffix('.recovery.tmp')
        with temporary.open('wb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return (blob, state)
