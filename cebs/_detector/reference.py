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
from cebs.calibration import NumericalSources, adapt_recurrence, cell_ranks, choose_event, cross_rank, dump, event_scores, fit_mask, held_risk, normal_routing, physical_cut, physical_ranks, rank, safe_json, supervision_scores
from cebs.reference import REQUIRED_FULL_PIPELINE_STAGES, fit_readout, nested_normality, require_complete_pipeline, require_physical_disjoint
from cebs._detector.predictor import ProbePredictor
from cebs._detector.io import ahash, digest, read_npz
from cebs._detector.pipeline import EvidencePipeline

class ReferenceEvidence(EvidencePipeline):
    """Only reuse the numerical full-response generator, not nested-fit heads."""

    def __init__(self, root, folder, case, spec, ref, clean, checkpoint, cfg, batch):
        (self.root, self.folder, self.case) = (root, folder, case)
        (self.spec, self.ref, self.poison) = (spec, ref, clean)
        self.checkpoint = self.channel_checkpoint = checkpoint
        (self.cfg, self.batch) = (cfg, batch)
        self.outer = -1
        (self.blocks, self.starts) = (ref['block_ids'], ref['starts'])
        self.fit = np.ones(len(self.blocks), bool)
        (self.n, self.channels) = (len(clean['series']) - 23, clean['series'].shape[1])
        self.stages = {}
        self.started = time.monotonic()
        folder.mkdir(parents=True, exist_ok=True)
        self.supports = np.asarray([(s, length) for length in (2, 4, 6) for s in range(1, 13 - length)], np.int64)
        assert self.supports.shape == (24, 2)
        (self.m, self.std) = (float(ref['mean']), float(ref['std']))
        ix = np.arange(self.n)[:, None]
        self.x = ((clean['series'][ix + np.arange(12)].astype(np.float64) - self.m) / self.std).astype(np.float32)
        self.y = clean['series'][ix + 12 + np.arange(12)].astype(np.float32)
        self.marks = clean['marks'][ix + np.arange(12)].astype(np.float32)
        from cebs._numerics.support_alignment import robust_location_scale, horizon_excess_fit, binding_cube, template_strength, conditional_permutation
        from cebs._numerics.support_clusters import discover_trajectory_cluster, prune_nodes
        src = NumericalSources(root, folder / 'numerical_sources')
        runner = src.load('support_selection.py', ['active_nodes'], dict(discover_trajectory_cluster=discover_trajectory_cluster, prune_nodes=prune_nodes))
        self.helpers = SimpleNamespace(**locals())

    def status(self, stage, **extra):
        record = dict(state='running', case=self.case, stage=stage, elapsed_seconds=time.monotonic() - self.started, no_attack=True, victim_training=False, full_pool_fit=True, **extra)
        dump(self.folder / 'status.json', record)
        print(json.dumps(record), flush=True)

    def predictor(self, checkpoint, label):
        import torch
        from cebs.rng import seed_all
        from models.predictors.FEDformer import Model

        def ns(value):
            return SimpleNamespace(**{k: ns(v) for (k, v) in value.items()}) if isinstance(value, dict) else value

        class Forecaster(torch.nn.Module):

            def __init__(self, config):
                super().__init__()
                self.net = Model(ns(config))
                self.to('cuda')

            def forward(self, x, marks):
                return self.net(x, marks, x.new_zeros(len(x), 12, x.shape[2]), None)

        def linear_features(seasonal, trend):
            value = torch.nn.functional.pad(trend, (1, 1), mode='circular').unfold(-1, 3, 1)
            value = value.permute(0, 2, 1, 3).flatten(2)
            return torch.cat([seasonal, value], -1)[:, -12:]
        zero = dict(full_weights=np.zeros((2048, self.channels), np.float32), full_bias=np.zeros(self.channels, np.float32), fold_weights=np.zeros((12, 2048, self.channels), np.float32), fold_bias=np.zeros((12, self.channels), np.float32))
        pred = ProbePredictor(checkpoint, zero, self.m, self.std, self.batch, self.spec['dataset'] == 'PEMS03', (torch, seed_all, Forecaster, linear_features))
        target = self.folder / f'{label}_a0_full_and_loo.npz'
        if target.exists():
            meta = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
            if digest(target) != meta['sha256']:
                raise RuntimeError('Readout cache hash changed')
            a0 = read_npz(target)
        else:
            (ff, bb) = ([], [])
            with torch.inference_mode():
                for lo in range(0, len(self.blocks), self.batch):
                    xx = torch.as_tensor(self.ref['x'][lo:lo + self.batch], device='cuda')
                    mm = None if pred.pems else torch.as_tensor(self.ref['marks'][lo:lo + self.batch], device='cuda')
                    pred.model(xx, mm)
                    ff.append(linear_features(pred.capture['s'], pred.capture['t']).cpu().numpy())
                    bb.append(xx.mean(1)[:, None, :].expand(-1, 12, -1).cpu().numpy())
            (ff, bb) = (np.concatenate(ff).astype(np.float64), np.concatenate(bb).astype(np.float64))
            (weights, bias, records) = ([], [], [])
            for block in [None] + list(range(12)):
                keep = np.ones(len(self.blocks), bool) if block is None else self.blocks != block
                (w, b) = fit_readout(ff, bb, self.ref['y'], keep, self.m, self.std, 10.0)
                weights.append(w)
                bias.append(b)
                records.append(dict(excluded_block=block, fit_blocks=np.unique(self.blocks[keep]).tolist(), fit_rows=int(keep.sum()), fit_starts_sha256=ahash(self.starts[keep])))
                self.status('fit_clean_reference_readouts', heads_complete=len(weights), heads_total=13)
            a0 = dict(full_weights=weights[0], full_bias=bias[0], fold_weights=np.stack(weights[1:]), fold_bias=np.stack(bias[1:]))
            np.savez_compressed(target, **a0)
            dump(target.with_suffix('.json'), dict(sha256=digest(target), heads=records, checkpoint_sha256=digest(checkpoint), poisoned_cache_reused=False))
            del ff, bb
            gc.collect()
        pred.weights = torch.as_tensor(a0['full_weights'], device='cuda')
        pred.bias = torch.as_tensor(a0['full_bias'], device='cuda')
        pred.fold_weights = torch.as_tensor(a0['fold_weights'], device='cuda')
        pred.fold_bias = torch.as_tensor(a0['fold_bias'], device='cuda')
        (f, g, _) = pred.predict(self.ref['x'][:4], self.ref['marks'][:4], self.blocks[:4])
        direct = pred.direct_reference_check(self.ref['x'][:4], self.ref['marks'][:4])
        np.testing.assert_allclose(g, direct, rtol=1e-05, atol=0.0001)
        dump(self.folder / f'{label}_direct_readout_check.json', dict(max_abs_error=float(np.abs(g - direct).max()), passed=True))
        return pred
