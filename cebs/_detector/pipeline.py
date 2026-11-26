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

def numerical_helpers(root, folder, outer):
    src = NumericalSources(root, folder / 'numerical_sources')
    from cebs._numerics.input_selection import node_evidence, soft_event_mask
    from cebs._numerics.support_clusters import discover_trajectory_cluster, prune_nodes
    from cebs._numerics.support_alignment import robust_location_scale, horizon_excess_fit, binding_cube, template_strength, conditional_permutation
    runner = src.load('support_selection.py', ['active_nodes'], dict(discover_trajectory_cluster=discover_trajectory_cluster, prune_nodes=prune_nodes))
    core_names = ['protocol', 'independent_positions', 'motif_rows', 'percent_flags', 'precursor', 'reconstruct_trusted', 'bh', 'window_mask']
    core = src.load('recurrence_patterns.py', core_names, dict(HISTORY=12, HORIZON=12, LENGTHS=tuple(range(3, 13)), PHASE=96, FOLDS=4, MIN_FIT_EVENTS=8, MIN_FIT_BLOCKS=3, MIN_QUERY_EVENTS=3, EVENT_GAP=24, BIN_WIDTH=0.1, MIN_FEATURE_RMS=0.5, PERMUTATIONS=999, FDR=0.05, TRUSTED_RISK=0.05))
    cross = src.load('recurrence_matching.py', ['row_keys', 'match_groups', 'corridor_mask', 'threshold_boundaries', 'freeze'], dict(core=core, OUT=folder / 'recurrence', CORRECTED_BOUNDARIES=True, OUTER_HELD=outer), adapt_recurrence)
    bg = src.load('recurrence_support.py', ['support_bounds', 'background_denominator', 'stamp', 'ControlledVariants'], dict(exp=cross, VARIANTS=('background_only', 'support_only', 'background_support')))
    joint = src.load('supervision_scores.py', ['future_scores', 'choose_joint_threshold'])
    factor = src.load('label_projection.py', ['mapped_recurrence', 'mapped_trusted_recurrence'], dict(HISTORY=12, HORIZON=12, BLOCKS=12))
    channel = src.load('channel_scores.py', ['top_mean'])
    physical = src.load('physical_coverage.py', ['physical_coverage'])
    novelty = src.load('label_novelty.py', ['strict_novelty'], dict(RADIUS=11))
    normal = src.load('normality.py', ['features', 'scaled', 'knn_distance'], dict(K=5))
    top = src.load('window_scores.py', ['top_mean_rows'], dict(TOP_FRACTION=0.1))
    event = src.load('event_scores.py', ['robust_parameters', 'evidence_logit', 'pooled_lag_one_correlation', 'normalized_event_scores', 'finish_event'], dict(BLOCKS=12))
    return SimpleNamespace(**locals())

class EvidencePipeline:

    def input_responses(self):
        from cebs._numerics.input_response import shrink_support, response_score
        from cebs._numerics.donors import support_knn_reconstruct, amplitude_scale
        pred = self.predictor(self.checkpoint, 'input')
        try:
            original = self.folder / 'original_predictions.npz'
            fixed = self.folder.parent / 'fixed_infected_predictions.npz'
            if original.exists():
                value = read_npz(original)
                (tf, tg, to, pg) = (value[k] for k in ('tf', 'tg', 'to', 'pg'))
                fixed_meta = json.loads(fixed.with_suffix('.json').read_text(encoding='utf-8'))
                if digest(fixed) != fixed_meta['sha256']:
                    raise RuntimeError('Shared frozen-backbone prediction cache changed')
                pf = read_npz(fixed, ['pf'])['pf']
            else:
                (tf, tg, to) = pred.predict(self.ref['x'], self.ref['marks'], self.blocks)
                (pf, pg, _) = pred.predict(self.x, self.marks, np.full(self.n, self.outer))
                if not fixed.exists():
                    np.savez_compressed(fixed, pf=pf)
                    dump(fixed.with_suffix('.json'), dict(sha256=digest(fixed), checkpoint_sha256=digest(self.checkpoint), reference_independent=True, generated_in_this_audit=True))
                np.savez_compressed(original, tf=tf, tg=tg, to=to, pg=pg)
            rows = []
            for (si, (s, length)) in enumerate(self.supports):
                self.status('full_input_responses', support=si + 1, total_supports=24)
                shard = self.folder / f'input_support_{si:02d}.npz'
                if shard.exists():
                    provenance = json.loads((self.folder / f'input_support_{si:02d}_provenance.json').read_text(encoding='utf-8'))
                    if digest(shard) != provenance['artifact_sha256']:
                        raise RuntimeError(f'Support cache changed: {shard}')
                    rows.append(read_npz(shard))
                    continue
                tick = time.monotonic()
                (s, length) = (int(s), int(length))
                tx = self.ref['x']
                tm = self.ref['marks']
                tsx = shrink_support(tx, s, length, 0.0625)
                (fs, gs, gos) = pred.predict(tsx, tm, self.blocks)
                (matched, full_matched) = (np.empty_like(tx), np.empty_like(tx))
                distance = np.empty(len(tx))
                provenance = []

                def reconstruction_and_amplitude(fit):
                    fit_ids = np.flatnonzero(fit)
                    rr = np.empty_like(tx[fit])
                    for b in np.unique(self.blocks[fit]):
                        query = fit & (self.blocks == b)
                        donor = fit & (self.blocks != b)
                        (raw, _, _, _) = support_knn_reconstruct(tx[query], tx[donor], s, length, 5)
                        rr[np.isin(fit_ids, np.flatnonzero(query))] = raw
                    return amplitude_scale(tx[fit], rr, tsx[fit])
                outer_amp = reconstruction_and_amplitude(self.fit)
                for query in range(12):
                    fit = fit_mask(self.blocks, self.outer, query)
                    test = self.blocks == query
                    amp = outer_amp if query == self.outer else reconstruction_and_amplitude(fit)
                    (rr, dist, _, _) = support_knn_reconstruct(tx[test], tx[fit], s, length, 5)
                    delta = rr.astype(np.float64) - tx[test]
                    matched[test] = tx[test].astype(np.float64) + amp * delta
                    full_matched[test] = tx[test].astype(np.float64) + outer_amp * delta
                    distance[test] = dist
                    provenance.append(dict(query_block=query, donor_blocks=np.unique(self.blocks[fit]).tolist(), amplitude=amp))
                (fk, _, gok) = pred.predict(matched, tm, self.blocks)
                (fdk, gdk, _) = pred.predict(full_matched, tm, self.blocks)
                (ps, pk, pd) = (np.empty(self.n), np.empty(self.n), np.empty(self.n))
                for lo in range(0, self.n, 512):
                    hi = min(self.n, lo + 512)
                    (xx, mm) = (self.x[lo:hi], self.marks[lo:hi])
                    sx = shrink_support(xx, s, length, 0.0625)
                    (rr, dist, _, _) = support_knn_reconstruct(xx, tx[self.fit], s, length, 5)
                    kx = (xx.astype(np.float64) + outer_amp * (rr.astype(np.float64) - xx)).astype(np.float32)
                    block = np.full(hi - lo, self.outer)
                    (pfs, pgs, _) = pred.predict(sx, mm, block)
                    (pfk, pgk, _) = pred.predict(kx, mm, block)
                    ps[lo:hi] = response_score(pf[lo:hi], pg[lo:hi], pfs, pgs, self.std)
                    pk[lo:hi] = response_score(pf[lo:hi], pg[lo:hi], pfk, pgk, self.std)
                    pd[lo:hi] = dist
                row = dict(trusted_shrink=response_score(tf, to, fs, gos, self.std), trusted_knn=response_score(tf, to, fk, gok, self.std), trusted_distance=distance, discovery_trusted_shrink=response_score(tf, tg, fs, gs, self.std), discovery_trusted_knn=response_score(tf, tg, fdk, gdk, self.std), poison_shrink=ps, poison_knn=pk, poison_distance=pd)
                np.savez_compressed(shard, **row)
                dump(self.folder / f'input_support_{si:02d}_provenance.json', dict(donors=provenance, global_amplitude_fit_blocks=np.unique(self.blocks[self.fit]).tolist(), global_amplitude=outer_amp, seconds=time.monotonic() - tick, artifact_sha256=digest(shard)))
                rows.append(row)
                self.status('input_support_complete', support=si + 1, total_supports=24, recent_support_seconds=time.monotonic() - tick)
            dual = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
            dual['supports'] = self.supports
            h = self.helpers
            (sm, ss) = h.robust_location_scale(dual['discovery_trusted_shrink'][:, self.fit], axis=1)
            (km, ks) = h.robust_location_scale(dual['discovery_trusted_knn'][:, self.fit], axis=1)
            rel = np.quantile(dual['trusted_distance'][:, self.fit], 0.95, axis=1)
            ins = np.minimum((dual['poison_shrink'] - sm[:, None]) / ss[:, None], (dual['poison_knn'] - km[:, None]) / ks[:, None])
            ins[dual['poison_distance'] > rel[:, None]] = -np.inf
            th = h.horizon_excess_fit(tf[self.fit], tg[self.fit], self.ref['y'][self.fit], self.std, 0.05)
            (hm, hs) = h.robust_location_scale(th, axis=1)
            ph = h.horizon_excess_fit(pf, pg, self.y, self.std, 0.05)
            hz = (ph - hm[:, None]) / hs[:, None]
            observed = h.template_strength(h.binding_cube(ins, hz), 1.5, 6.0)
            rng = np.random.default_rng(260912)
            null = np.asarray([h.template_strength(h.binding_cube(ins, hz[:, h.conditional_permutation(np.arange(self.n), rng, self.cfg['period'])]), 1.5, 6.0) for _ in range(40)])
            (nm, ns) = (null.mean(0), np.maximum(null.std(0, ddof=1), 1e-08))
            dual.update(observed_strength=observed, null_mean=nm, template_z=(observed - nm) / ns)
            np.savez_compressed(self.folder / 'dual_fresh.npz', **dual)
            self.stages['input_a0_and_knn'] = 'complete'
            return dual
        finally:
            pred.close()
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def evidence(self, dual):
        from cebs._numerics.input_selection import node_evidence
        (ts, tk, td) = (dual[k] for k in ('trusted_shrink', 'trusted_knn', 'trusted_distance'))
        (ps, pk, pd) = (dual[k] for k in ('poison_shrink', 'poison_knn', 'poison_distance'))
        (tev, pev) = (np.zeros(ts.shape), np.zeros(ps.shape))
        for s in range(len(ts)):
            for b in range(12):
                fit = fit_mask(self.blocks, self.outer, b)
                test = self.blocks == b
                tev[s, test] = node_evidence(rank(ts[s, fit], ts[s, test], False), rank(tk[s, fit], tk[s, test], False), td[s, test] <= np.quantile(td[s, fit], 0.95), 0.95, 0.95, 1.0)
            pev[s] = node_evidence(rank(ts[s, self.fit], ps[s], False), rank(tk[s, self.fit], pk[s], False), pd[s] <= np.quantile(td[s, self.fit], 0.95), 0.95, 0.95, 1.0)
        return (tev, pev)

    def masks_at_cut(self, tev, pev, supports, cut, bounds):
        from cebs._numerics.input_selection import soft_event_mask
        (thard, hard) = (np.zeros(len(self.blocks), bool), np.zeros(self.n, bool))
        if len(supports) and cut is not None:
            for b in range(12):
                test = self.blocks == b
                thard[test] = soft_event_mask(tev[:, test], supports, cut, 5, bounds)[0]
            hard = soft_event_mask(pev, supports, cut, 5, bounds)[0]
        return (thard, hard)

    def path_a(self, dual):
        from cebs._numerics.temporal_projection import corridor_bounds, response_offsets
        self.status('rediscover_nodes_and_path_a')
        try:
            (active, cluster) = self.helpers.runner.active_nodes(self.cfg, dual)
        except RuntimeError as exc:
            if 'cluster' not in str(exc).lower():
                raise
            (active, cluster) = ([], dict(abstain=str(exc)))
        ids = np.asarray([r['support_index'] for r in active], np.int64)
        supports = dual['supports'][ids]
        bounds = corridor_bounds(response_offsets(active, 12), 12, 12, 1) if active else (-11, 0)
        sub = {k: dual[k][ids] for k in ('trusted_shrink', 'trusted_knn', 'trusted_distance', 'poison_shrink', 'poison_knn', 'poison_distance')}
        (tev, pev) = self.evidence(sub)
        path = []
        for cut in np.arange(0.5, len(ids) * 2 + 0.125, 0.25):
            (tb, _) = self.masks_at_cut(tev, pev, supports, cut, bounds)
            path.append(dict(threshold=float(cut), empirical_rate=float(tb[self.fit].mean())))
        chosen = next((r['threshold'] for r in path if r['empirical_rate'] <= 0.05), None)
        (tb, base) = self.masks_at_cut(tev, pev, supports, chosen, bounds)
        self.stages['path_a_discovery_and_calibration'] = 'complete'
        dump(self.folder / 'path_a.json', dict(active_nodes=active, cluster=cluster, corridor_bounds=bounds, selected_threshold=chosen, risk_path=path, fit_blocks=np.unique(self.blocks[self.fit]).tolist(), selected_fit_risk=float(tb[self.fit].mean()), empty_node_abstention=not bool(active)))
        np.savez_compressed(self.folder / 'path_a_fresh.npz', supports=supports, fold_evidence=tev, poison_evidence=pev, hard=base, trusted_hard=tb)
        return dict(active=active, ids=ids, supports=supports, bounds=bounds, tev=tev, pev=pev, cut=chosen, path=path, hard=base, trusted_hard=tb)

    def recurrence(self, dual, path):
        self.status('fresh_recurrence_pairing_and_stable_core')
        (h, audit) = (self.helpers, self)
        data = dict(train=self.poison['series'].astype(np.float64), trusted=self.ref, supports=path['supports'], trusted_scores=dual['trusted_shrink'][path['ids']], poison_scores=dual['poison_shrink'][path['ids']], baseline=path['hard'], bounds=path['bounds'], period=self.cfg['period'], poison_path=self.spec['poison'], lineage='Fresh full nested outer-block reconstruction')

        class Observer(h.bg.ControlledVariants):

            def begin(self, data, edges):
                self.raws = h.core.reconstruct_trusted(data['trusted'])
                self.point = {v: np.full(data['train'].shape, -np.inf) for v in h.bg.VARIANTS}
                self.tpoint = {v: [np.full(r.shape, -np.inf) for r in self.raws] for v in h.bg.VARIANTS}

            def finish(self, saved, decisions, dest):
                score = h.joint.future_scores(self.point['support_only'])
                ts = np.concatenate([h.joint.future_scores(r) for r in self.tpoint['support_only']])
                tb = path['trusted_hard']
                (cut, risk) = h.joint.choose_joint_threshold(tb[audit.fit], ts[audit.fit])
                cut = float('inf') if cut is None else cut
                cuts = []
                for b in np.unique(audit.blocks[audit.fit]):
                    fit = fit_mask(audit.blocks, audit.outer, b)
                    (c, _) = h.joint.choose_joint_threshold(tb[fit], ts[fit])
                    if c is not None and np.isfinite(c):
                        cuts.append(c)
                stable_cut = max(cut, float(np.quantile(cuts, 0.75, method='higher'))) if len(cuts) >= 6 else float('inf')
                self.result = dict(hard=path['hard'] | (score >= stable_cut), trusted_hard=tb | (ts >= stable_cut), train_block_edges=saved['train_block_edges'])
                dump(audit.folder / 'stable_core.json', dict(joint_cut=cut, stable_cut=stable_cut, feasible_inner_folds=len(cuts), joint_fit_risk=risk, fit_blocks=np.unique(audit.blocks[audit.fit]).tolist(), trusted_fit_risk=float(self.result['trusted_hard'][audit.fit].mean())))
                np.savez_compressed(audit.folder / 'stable_core_fresh.npz', **self.result)
        observer = Observer(self.case)
        recpath = self.folder / 'recurrence' / self.case / 'frozen_preoracle.npz'
        if recpath.exists() and (self.folder / 'stable_core_fresh.npz').exists():
            result = read_npz(self.folder / 'stable_core_fresh.npz')
        else:
            if recpath.exists():
                raise RuntimeError('Incomplete recurrence transaction; use a new run directory')
            h.cross.prepare = lambda _: data
            h.cross.freeze.__globals__['prepare'] = h.cross.prepare
            h.cross.freeze(self.case, observer)
            result = observer.result
        rec = read_npz(recpath)
        meta = json.loads((recpath.parent / 'preoracle.json').read_text(encoding='utf-8'))
        rec['trusted_threshold'] = np.asarray(meta['decisions']['recurrence_only']['threshold'])
        self.stages['recurrence_scale_pairing_and_stable_core'] = 'complete'
        dump(self.folder / 'recurrence_outer_provenance.json', dict(outer_excluded=self.outer, folds=[dict(phase_fold=f, fit_trusted_blocks=np.flatnonzero((np.arange(12) % 4 != f) & (np.arange(12) != self.outer)).tolist()) for f in range(4)], motif_threshold_fit_blocks=np.flatnonzero(np.arange(12) != self.outer).tolist(), monte_carlo_draws=999, historical_masks_reused=False))
        return (rec, result)

    def local_masks(self, rec, stable, path):
        self.status('channel_gate_and_local_mask_reconstruction')
        h = self.helpers
        (hard, thard) = (stable['hard'], stable['trusted_hard'])
        pr = h.factor.mapped_recurrence(rec['recurrence_score'], self.n)
        tr = h.factor.mapped_trusted_recurrence(rec['trusted_recurrence_score'], self.blocks, self.channels)
        cut = float(rec['trusted_threshold'])
        existing = (pr >= cut) & ~hard[:, None, None]
        texisting = (tr >= cut) & ~thard[:, None, None]
        closure_bounds = (min(path['bounds'][0], -11), max(path['bounds'][1], 0))
        closecut = None
        for row in path['path']:
            (tb, _) = self.masks_at_cut(path['tev'], path['pev'], path['supports'], row['threshold'], closure_bounds)
            if tb[self.fit].mean() <= 0.05:
                closecut = row['threshold']
                break
        (tclose, close) = self.masks_at_cut(path['tev'], path['pev'], path['supports'], closecut, closure_bounds)
        candidate = close & ~hard
        t_coupling = np.zeros((len(self.blocks), 12, self.channels), np.float32)
        p_coupling = np.full((self.n, 12, self.channels), -np.inf, np.float32)
        if candidate.any() and len(path['supports']):
            target = self.folder / 'channel_coupling_fresh.npz'
            if target.exists():
                value = read_npz(target)
                (t_coupling, p_coupling) = (value['trusted'], value['poison'])
            else:
                from cebs._numerics.input_response import shrink_support
                pred = self.predictor(self.channel_checkpoint, 'channel')
                try:
                    (tf, _, tg) = pred.predict(self.ref['x'], self.ref['marks'], self.blocks)
                    ids = np.flatnonzero(candidate)
                    (pf, pg, _) = pred.predict(self.x[ids], self.marks[ids], np.full(len(ids), self.outer))
                    pbest = np.zeros_like(pf)
                    for (s, length) in path['supports']:
                        (fs, _, gs) = pred.predict(shrink_support(self.ref['x'], int(s), int(length), 0.0625), self.ref['marks'], self.blocks)
                        t_coupling = np.maximum(t_coupling, np.abs(tf - fs - (tg - gs)))
                        (pfs, pgs, _) = pred.predict(shrink_support(self.x[ids], int(s), int(length), 0.0625), self.marks[ids], np.full(len(ids), self.outer))
                        pbest = np.maximum(pbest, np.abs(pf - pfs - (pg - pgs)))
                    p_coupling[ids] = pbest
                    np.savez_compressed(target, trusted=t_coupling, poison=p_coupling)
                finally:
                    pred.close()
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
        (tcp, pcp) = cell_ranks(t_coupling, p_coupling, self.blocks, self.outer)
        bgids = np.flatnonzero(np.arange(12) != self.outer)
        ref_gate = np.asarray([h.channel.top_mean(pcp[candidate, :, c]) > max((h.channel.top_mean(tcp[self.blocks == b, :, c]) for b in bgids)) if candidate.any() else False for c in range(self.channels)])
        rec_gate = np.asarray([h.channel.top_mean(rec['recurrence_score'][:, c]) > max((h.channel.top_mean(rec['trusted_recurrence_score'][b, :, c]) for b in bgids)) for c in range(self.channels)])
        confirmed = ref_gate & rec_gate
        (tp, pp) = physical_ranks(rec['trusted_recurrence_score'][:, 12:], rec['recurrence_score'][12:], self.outer)
        (pcut, stability) = physical_cut(tp, confirmed, self.outer) if confirmed.any() else (float('inf'), dict(feasible_folds=0))
        selected = confirmed[None, :] & (pp >= pcut)
        (removed, total) = h.physical.physical_coverage(hard, existing)
        actionable = (removed > 0) & (removed < total) | h.novelty.strict_novelty(removed > 0)
        selected &= actionable
        local = existing | selected[np.arange(self.n)[:, None] + np.arange(12)]
        tlocal = texisting.copy()
        for b in range(12):
            ids = np.flatnonzero(self.blocks == b)
            (removed, total) = h.physical.physical_coverage(thard[ids], texisting[ids])
            sel = confirmed[None, :] & (tp[b] >= pcut)
            sel &= (removed > 0) & (removed < total) | h.novelty.strict_novelty(removed > 0)
            tlocal[ids] |= sel[np.arange(len(ids))[:, None] + np.arange(12)]
        np.savez_compressed(self.folder / 'local_union_fresh.npz', local=local, trusted_local=tlocal, existing=existing, trusted_existing=texisting)
        dump(self.folder / 'channel_gate.json', dict(confirmed_channels=np.flatnonzero(confirmed).tolist(), reference_gate=ref_gate, recurrence_gate=rec_gate, physical_cut=pcut, stability=stability, fit_trusted_blocks=bgids.tolist(), channel_checkpoint=str(self.channel_checkpoint), closure_cut=closecut, closure_candidates=int(candidate.sum())))
        self.stages['channel_and_local_masks'] = 'complete'
        return (local, tlocal, pr, tr)
