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

class ProbePredictor:
    """One frozen backbone; cached linear readouts evaluated from its features."""

    def __init__(self, checkpoint, a0, mean, std, batch, pems, modules):
        (torch, seed_all, Forecaster, linear_features) = modules
        (self.torch, self.linear_features) = (torch, linear_features)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        seed_all(1)
        self.model = Forecaster(state['model_config'])
        self.model.net.load_state_dict(state['model'])
        self.model.eval()
        (self.mean, self.std, self.batch, self.pems) = (float(mean), float(std), batch, pems)
        self.weights = torch.as_tensor(a0['full_weights'], device='cuda')
        self.bias = torch.as_tensor(a0['full_bias'], device='cuda')
        self.fold_weights = torch.as_tensor(a0['fold_weights'], device='cuda')
        self.fold_bias = torch.as_tensor(a0['fold_bias'], device='cuda')
        self.capture = {}

        def hook(name):
            return lambda module, args: self.capture.__setitem__(name, args[0].detach())
        self.handles = [self.model.net.decoder.projection.register_forward_pre_hook(hook('s')), self.model.net.decoder.layers[0].projection.register_forward_pre_hook(hook('t'))]
        self.meta = {k: state[k] for k in ('epoch', 'seed', 'initialization_seed', 'model_config') if k in state}

    def predict(self, x, marks, blocks):
        torch = self.torch
        rows = [[], [], []]
        with torch.inference_mode():
            for lo in range(0, len(x), self.batch):
                hi = min(lo + self.batch, len(x))
                xx = torch.as_tensor(x[lo:hi], device='cuda', dtype=torch.float32)
                mm = None if self.pems else torch.as_tensor(marks[lo:hi], device='cuda', dtype=torch.float32)
                f = self.model(xx, mm)
                base = xx.mean(1)[:, None, :]
                seasonal = self.capture['s']
                trend = self.capture['t']

                def head(w, b):
                    ss = torch.nn.functional.linear(seasonal, w[:512].T, b)
                    tt = torch.nn.functional.conv1d(torch.nn.functional.pad(trend, (1, 1), mode='circular'), w[512:].T.reshape(-1, 512, 3)).transpose(1, 2)
                    return (base + tt + ss)[:, -12:]
                g = head(self.weights, self.bias)
                bb = torch.as_tensor(blocks[lo:hi], device='cuda', dtype=torch.long)
                goof = torch.empty_like(g)
                for block in torch.unique(bb):
                    selected = bb == block
                    goof[selected] = head(self.fold_weights[block], self.fold_bias[block])[selected]
                for (j, value) in enumerate((f, g, goof)):
                    rows[j].append((value * self.std + self.mean).cpu().numpy())
        return tuple((np.concatenate(r).astype(np.float32) for r in rows))

    def direct_reference_check(self, x, marks):
        torch = self.torch
        sw = self.model.net.decoder.projection.weight.detach().clone()
        sb = self.model.net.decoder.projection.bias.detach().clone()
        tw = self.model.net.decoder.layers[0].projection.weight.detach().clone()
        try:
            with torch.no_grad():
                self.model.net.decoder.projection.weight.copy_(self.weights[:512].T)
                self.model.net.decoder.projection.bias.copy_(self.bias)
                self.model.net.decoder.layers[0].projection.weight.copy_(self.weights[512:].T.reshape(-1, 512, 3))
                xx = torch.as_tensor(x, device='cuda')
                mm = None if self.pems else torch.as_tensor(marks, device='cuda')
                raw = (self.model(xx, mm) * self.std + self.mean).cpu().numpy()
        finally:
            with torch.no_grad():
                self.model.net.decoder.projection.weight.copy_(sw)
                self.model.net.decoder.projection.bias.copy_(sb)
                self.model.net.decoder.layers[0].projection.weight.copy_(tw)
        return raw

    def close(self):
        for h in self.handles:
            h.remove()
        del self.model, self.weights, self.bias, self.fold_weights, self.fold_bias
