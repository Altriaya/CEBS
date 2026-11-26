"""CEBS core numerical implementation."""
from __future__ import annotations

import ast
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cebs.events import select_cut

def fit_mask(blocks, outer, query=None):
    return (np.asarray(blocks) != outer) & (True if query is None else np.asarray(blocks) != query)

def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return str(value)
    return value

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

class NumericalSources:
    """Execute named numerical definitions, not archived training entrypoints.

    Selected source and the narrow outer-block adaptation are saved verbatim.
    No import shim or monkey-patch changes the originals on disk.
    """

    def __init__(self, root, destination):
        (self.root, self.destination) = (Path(root), Path(destination))
        self.records = {}

    def load(self, relative, names, context=None, transform=None):
        from cebs.numerics import bootstrap, numerical_source
        bootstrap()
        path = numerical_source(relative)
        source = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        wanted = set(names)
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
        if {n.name for n in nodes} != wanted:
            raise ValueError(f'Numerical definitions missing: {relative}: {wanted - {n.name for n in nodes}}')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
        if transform is not None:
            module = transform(module)
        ast.fix_missing_locations(module)
        adapted = ast.unparse(module)
        namespace = dict(np=np, time=time, Path=Path, hashlib=hashlib, json=json, defaultdict=defaultdict, ROOT=self.root, dump=dump)
        namespace.update(context or {})
        exec(compile(module, str(path), 'exec'), namespace)
        key = relative.replace('/', '__')
        self.destination.mkdir(parents=True, exist_ok=True)
        snapshot = self.destination / key
        if snapshot.exists() and snapshot.read_text(encoding='utf-8') != adapted:
            raise RuntimeError(f'Numerical source changed on resume: {relative}')
        snapshot.write_text(adapted, encoding='utf-8')
        self.records[relative] = dict(original_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(), original_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), selected=list(names), adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest())
        return SimpleNamespace(**namespace)

    def verify_originals(self):
        from cebs.numerics import numerical_source
        for (relative, record) in self.records.items():
            if hashlib.sha256(numerical_source(relative).read_bytes()).hexdigest() != record['original_file_sha256']:
                raise RuntimeError(f'Numerical source changed: {relative}')

def adapt_recurrence(module):
    """Only replace trusted fitting populations and export newly built scores."""

    class Adapt(ast.NodeTransformer):

        def __init__(self):
            self.counts = dict(tfits=0, threshold=0, scores=0)

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if any((isinstance(t, ast.Name) and t.id == 'tfits' for t in node.targets)):
                expected = 'np.flatnonzero(np.arange(12) % 4 != fold)'
                if ast.unparse(node.value) != expected:
                    raise RuntimeError('Unexpected recurrence fit-block source')
                node.value = ast.parse('np.flatnonzero((np.arange(12) % 4 != fold) & (np.arange(12) != OUTER_HELD))', mode='eval').body
                self.counts['tfits'] += 1
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'threshold_boundaries' and (len(node.args) == 1):
                if ast.unparse(node.args[0]) == 'tpoint[variant]':
                    node.args[0] = ast.parse('[s for b, s in enumerate(tpoint[variant]) if b != OUTER_HELD]', mode='eval').body
                    self.counts['threshold'] += 1
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'np.savez_compressed':
                if any((k.arg is None and ast.unparse(k.value) == 'saved' for k in node.keywords)):
                    node.keywords += [ast.keyword(arg='recurrence_score', value=ast.parse("point['recurrence_only']", mode='eval').body), ast.keyword(arg='trusted_recurrence_score', value=ast.parse("np.stack(tpoint['recurrence_only'])", mode='eval').body)]
                    self.counts['scores'] += 1
            return node
    adapter = Adapt()
    module = adapter.visit(module)
    if adapter.counts != dict(tfits=1, threshold=1, scores=1):
        raise RuntimeError(f'Unexpected recurrence adaptation: {adapter.counts}')
    return module

def rank(reference, query, finite_only=True, dtype=np.float64):
    reference = np.sort(np.asarray(reference, np.float64).reshape(-1))
    if not len(reference):
        raise ValueError('Empty reference is not a zero-risk reference')
    query = np.asarray(query, np.float64)
    out = np.zeros(query.shape, dtype)
    valid = np.isfinite(query) if finite_only else np.ones(query.shape, bool)
    out[valid] = (np.searchsorted(reference, query[valid], side='right') + 1) / (len(reference) + 1)
    return out

def cross_rank(trusted, poison, blocks, outer, finite_only=True, dtype=np.float64):
    result = np.zeros(np.shape(trusted), dtype)
    for query in np.unique(blocks):
        result[blocks == query] = rank(trusted[fit_mask(blocks, outer, query)], trusted[blocks == query], finite_only, dtype)
    return (result, rank(trusted[fit_mask(blocks, outer)], poison, finite_only, dtype))

def cell_ranks(trusted, poison, blocks, outer):
    (result, train) = (np.zeros(trusted.shape, np.float32), np.zeros(poison.shape, np.float32))
    for h in range(trusted.shape[1]):
        for c in range(trusted.shape[2]):
            (result[:, h, c], train[:, h, c]) = cross_rank(trusted[:, h, c], poison[:, h, c], blocks, outer, dtype=np.float32)
    return (result, train)

def physical_ranks(trusted, poison, outer):
    (result, train) = (np.zeros(trusted.shape, np.float32), np.zeros(poison.shape, np.float32))
    for c in range(trusted.shape[-1]):
        for query in range(12):
            result[query, :, c] = rank(trusted[fit_mask(np.arange(12), outer, query), :, c], trusted[query, :, c], dtype=np.float32)
        train[:, c] = rank(trusted[np.arange(12) != outer, :, c], poison[:, c], dtype=np.float32)
    return (result, train)

def physical_cut(score, admitted, outer):

    def choose(ids):
        values = score[ids][:, :, admitted].reshape(-1)
        for cut in np.unique(values[np.isfinite(values)]):
            if (values >= cut).mean() <= 0.05 + 1e-12:
                return float(cut)
        return float('inf')
    ids = np.flatnonzero(np.arange(12) != outer)
    pooled = choose(ids)
    folds = [choose(ids[ids != b]) for b in ids]
    feasible = [c for c in folds if np.isfinite(c)]
    cut = max(pooled, float(np.quantile(feasible, 0.75, method='higher'))) if len(feasible) >= 6 else float('inf')
    return (cut, dict(pooled=pooled, inner_cuts=folds, feasible_folds=len(feasible), fit_blocks=ids.tolist()))

def supervision_scores(traw, praw, blocks, outer, top_mean_rows):
    tp = np.zeros_like(traw, dtype=np.float64)
    pp = np.zeros_like(praw, dtype=np.float64)
    for c in range(praw.shape[1]):
        pp[:, c] = rank(traw[np.arange(12) != outer, :, c], praw[:, c])
        for query in range(12):
            tp[query, :, c] = rank(traw[fit_mask(np.arange(12), outer, query), :, c], traw[query, :, c])
    result = np.empty(len(blocks))
    for b in range(12):
        ids = np.flatnonzero(blocks == b)
        cells = tp[b, np.arange(len(ids))[:, None] + 12 + np.arange(12)]
        result[ids] = top_mean_rows(cells.reshape(len(ids), -1))
    n = len(praw) - 23
    train = np.empty(n)
    for lo in range(0, n, 256):
        hi = min(n, lo + 256)
        cells = pp[np.arange(lo, hi)[:, None] + 12 + np.arange(12)]
        train[lo:hi] = top_mean_rows(cells.reshape(hi - lo, -1))
    return (result, train)

def event_scores(helpers, trusted, train, tproposal, proposal, blocks, pblocks, starts, outer):
    fit = fit_mask(blocks, outer)
    tl = helpers.evidence_logit(trusted, int(fit.sum()))
    pl = helpers.evidence_logit(train, int(fit.sum()))
    tz = np.zeros(len(trusted))
    (rhos, records) = ({}, [])
    for query in range(12):
        allowed = fit_mask(blocks, outer, query)
        (location, scale) = helpers.robust_parameters(tl[allowed])
        z = (tl - location) / scale
        tz[blocks == query] = z[blocks == query]
        rho = helpers.pooled_lag_one_correlation(z, blocks, starts, allowed)
        rhos[query] = rho
        records.append(dict(query_block=query, fit_blocks=np.unique(blocks[allowed]).tolist(), location=location, scale=scale, rho=rho))
    (ts, tevents) = helpers.normalized_event_scores(tproposal, tz, blocks, starts, rhos)
    (location, scale) = helpers.robust_parameters(tl[fit])
    rho = helpers.pooled_lag_one_correlation((tl - location) / scale, blocks, starts, fit)
    (ps, pevents) = helpers.normalized_event_scores(proposal, (pl - location) / scale, pblocks, np.arange(len(train)), {b: rho for b in range(12)})
    return (ts, ps, dict(trusted_crossfit=records, train_location=location, train_scale=scale, train_rho=rho, finite_sample_size=int(fit.sum()), trusted_events=tevents, train_events=pevents))

def choose_event(score, base, proposal, blocks, outer):
    fit = fit_mask(blocks, outer)
    (cut, _, report) = select_cut(score[fit], base[fit], proposal[fit], blocks[fit], 0.05)
    final = base.copy()
    if cut is not None:
        final |= proposal & (score >= cut)
    report.update(fit_blocks=np.unique(blocks[fit]).tolist(), heuristic_not_distribution_free=True, candidate_risk=None if not proposal[fit].any() else float(final[fit & proposal].mean()), candidate_support=int(proposal[fit].sum()))
    return (cut, final, report)

def normal_routing(hard, count, normality, certificate_cut, promotion_cut, active):
    """Identical inference on poisoned training and held clean queries.

    A max-fit-count+1 cutoff prevents promotions on fitting rows, but not on
    held rows. Omitting held promotions would under-report outer risk.
    """
    if not active:
        return np.asarray(hard, bool).copy()
    rescue = (np.asarray(count) == 0) & (np.asarray(normality) <= certificate_cut)
    promotion = ~np.asarray(hard, bool) & (np.asarray(count) >= promotion_cut)
    return np.asarray(hard, bool) & ~rescue | promotion

def held_risk(hard, local, blocks, outer, proposal=None):
    held = blocks == outer
    if not held.any() or local.shape[:1] != hard.shape or np.any(local[hard]):
        raise ValueError('Invalid final held-out hard/local geometry')
    (h, l) = (hard[held], local[held])
    combined = h[:, None, None] | l
    result = dict(held_block=int(outer), windows=int(held.sum()), hard_deleted=int(h.sum()), hard_risk=float(h.mean()), local_label_cells=int(l.sum()), local_label_risk=float(l.mean()), total_label_loss=float(combined.mean()), local_window_rate=float(l.any(axis=(1, 2)).mean()), retained_label_local_risk=None if h.all() else float(l[~h].mean()))
    if proposal is not None:
        ids = held & proposal
        result.update(candidate_windows=int(ids.sum()), candidate_hard_risk=None if not ids.any() else float(hard[ids].mean()))
    return result
