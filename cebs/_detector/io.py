"""CEBS core numerical implementation."""
from __future__ import annotations

import ast
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
FORMAL = Path(__file__).resolve().parents[2]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for x in iter(lambda : stream.read(1 << 20), b''):
            h.update(x)
    return h.hexdigest()

def ahash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def read_npz(path: Path, keys=None) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}
