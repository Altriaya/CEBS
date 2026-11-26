"""CEBS core numerical implementation."""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from cebs.artifacts import ahash, dump, read, save_npz, sha
from cebs.numerics import bootstrap, NUMERICS, numerical_source
FORMAL = Path(__file__).resolve().parents[2]

def run_detector(root, folder, case, dataset, ref, observed, checkpoint, period, batch=64, channel_checkpoint=None):
    (root, folder, checkpoint) = (Path(root), Path(folder), Path(checkpoint))
    bootstrap()
    from cebs._detector.full_pool import FullPoolEvidence
    from cebs._detector.pipeline import EvidencePipeline
    from cebs.calibration import cross_rank, supervision_scores, event_scores, choose_event, normal_routing
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    if not np.array_equal(np.unique(ref['block_ids']), np.arange(12)):
        raise ValueError('The unchanged CEBS calibration requires twelve physical reference blocks')
    if 'marks' not in ref:
        ref = dict(ref, marks=np.zeros((len(ref['starts']), 12, 4), np.float32))
    if 'marks' not in observed:
        observed = dict(observed, marks=np.zeros((len(observed['series']), 4), np.float32))
    dependencies = sorted((FORMAL / 'cebs').rglob('*.py')) + [checkpoint]
    dependencies += sorted((FORMAL / 'models/predictors').rglob('*.py'))
    dependencies += sorted(NUMERICS.rglob('*.py'))
    dependencies.append(numerical_source('settings.json'))
    if channel_checkpoint is not None:
        dependencies.append(Path(channel_checkpoint))
    identity = dict(case=case, dataset=dataset, period=period, batch=batch, checkpoint_sha256=sha(checkpoint), reference_arrays={k: ahash(ref[k]) for k in ('x', 'y', 'starts', 'block_ids', 'mean', 'std')}, observed_arrays={k: ahash(observed[k]) for k in ('series', 'marks')}, sources={str(p): sha(p) for p in dependencies})
    identity['reference_marks_sha256'] = ahash(ref['marks'])
    identity['channel_checkpoint_sha256'] = sha(channel_checkpoint or checkpoint)
    manifest = folder / 'input_identity.json'
    if manifest.exists() and read(manifest) != identity:
        raise RuntimeError('Detector cache belongs to different data, proxy, references, or code')
    dump(manifest, identity)
    report_path = folder / 'complete.json'
    if report_path.exists():
        report = read(report_path)
        if sha(folder / 'final_masks.npz') != report['mask_file_sha256']:
            raise RuntimeError('Completed detector artifact changed')
        return report
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = read(numerical_source('settings.json'))
    common['period'] = period

    class NewAssetAudit(FullPoolEvidence):

        def status(self, stage, **extra):
            value = dict(state='running', case=case, stage=stage, seconds=time.monotonic() - self.started, fresh_detector=True, historical_masks_used=False, victim_training=False, **extra)
            dump(folder / 'status.json', value)
            print(json.dumps(value), flush=True)

        def finish(self, dual, rec, stable, path, local, tlocal, pr, tr):
            h = self.helpers
            self.status('normality_base_event_and_final_protection')
            tf = h.normal.features(self.ref['y'])
            normal = self.normal_helper.trusted_scores(tf, self.blocks)
            (query, donors) = h.normal.scaled(h.normal.features(self.y), tf)
            pnormal = h.normal.knn_distance(query, donors)
            rec_cut = float(rec['trusted_threshold'])
            count = (np.isfinite(pr) & (pr >= rec_cut)).sum((1, 2))
            tcount = (np.isfinite(tr) & (tr > rec_cut)).sum((1, 2))
            ncut = float(np.quantile(normal, 0.75))
            (cert, tcert) = ((count == 0) & (pnormal <= ncut), (tcount == 0) & (normal <= ncut))
            background = int(np.unique(self.blocks[tcount > 0]).size)
            (seed, tseed) = (stable['hard'], stable['trusted_hard'])
            active = background >= 3 and float((tseed & ~tcert).mean()) <= 0.05
            promotion = int(tcount.max()) + 1
            base = normal_routing(seed, count, pnormal, ncut, promotion, active)
            tb = normal_routing(tseed, tcount, normal, ncut, promotion, active)
            (st, sp) = supervision_scores(rec['trusted_recurrence_score'], rec['recurrence_score'], self.blocks, -1, h.top.top_mean_rows)
            proposal_cut = float(np.quantile(st, 0.95))
            (tproposal, proposal) = (~tb & (st >= proposal_cut), ~base & (sp >= proposal_cut))
            (ti, pi) = self.evidence(dual)
            (ts, ps) = cross_rank(st, sp, self.blocks, -1)
            (tip, pip) = cross_rank(ti.max(0), pi.max(0), self.blocks, -1)
            (tn, pn) = cross_rank(normal, pnormal, self.blocks, -1)
            (ta, pa) = (np.minimum(np.minimum(ts, tip), tn), np.minimum(np.minimum(ps, pip), pn))
            pblocks = np.minimum(np.searchsorted(stable['train_block_edges'][1:], np.arange(self.n), side='right'), 11)
            (te, pe, parameters) = event_scores(h.event, ta, pa, tproposal, proposal, self.blocks, pblocks, self.starts, -1)
            (cut, toriginal, selection) = choose_event(te, tb, tproposal, self.blocks, -1)
            added = np.zeros(self.n, bool) if cut is None else proposal & (pe >= cut)
            original = base | added
            hard = original & ~(cert & active)
            thard = toriginal & ~(tcert & active)
            (final_local, trusted_local) = (local & ~hard[:, None, None], tlocal & ~thard[:, None, None])
            if final_local[hard].any() or np.any(final_local[~hard].all((1, 2))):
                raise RuntimeError('Invalid final hard/local composition')
            save_npz(folder / 'final_masks.npz', hard_mask=hard, local_mask=final_local, trusted_hard=thard, trusted_local=trusted_local, seed_hard=seed, base_hard=base, before_final_protection=original, count=count, trusted_count=tcount, normality=pnormal, trusted_normality=normal, trusted_starts=self.starts, trusted_block_ids=self.blocks)
            dump(folder / 'event_parameters.json', parameters)
            return dict(state='complete_fresh_cebs', case=case, window_count=self.n, deleted_windows=int(hard.sum()), local_masked_cells=int(final_local.sum()), hard_sha256=ahash(hard), local_sha256=ahash(final_local), mask_file_sha256=sha(folder / 'final_masks.npz'), event_threshold=cut, event_selection=selection, normal_protection_active=bool(active), input_deleted_windows=int(path['hard'].sum()), seed_deleted_windows=int(seed.sum()), event_additions=int(added.sum()), released_windows=int((original & ~hard).sum()), active_nodes=path['active'], fresh_reference_heads=True, historical_masks_used=False, oracle_input=False, victim_training=False)
    spec = dict(dataset=dataset, poison=str(folder / 'observed_training.npz'))
    save_npz(folder / 'observed_training.npz', **observed)
    audit = NewAssetAudit(root, folder, case, spec, ref, observed, checkpoint, common, batch)
    audit.channel_checkpoint = Path(channel_checkpoint) if channel_checkpoint else checkpoint
    try:
        with threadpool_limits(limits=4):
            dual = audit.input_responses()
            (active, cluster) = audit.helpers.runner.active_nodes(common, dual)
            if not active:
                raise RuntimeError('CEBS discovery failure; no fallback or fabricated mask is permitted')
            path = EvidencePipeline.path_a(audit, dual)
            if path['active'] != active:
                raise RuntimeError('Discovery changed between identical calls')
            (rec, stable) = EvidencePipeline.recurrence(audit, dual, path)
            (local, tlocal, pr, tr) = EvidencePipeline.local_masks(audit, rec, stable, path)
            report = audit.finish(dual, rec, stable, path, local, tlocal, pr, tr)
        audit.helpers.src.verify_originals()
        audit.extra_src.verify_originals()
        if any((sha(p) != v for (p, v) in identity['sources'].items())):
            raise RuntimeError('Detector inputs or code changed while replaying')
        report['seconds'] = time.monotonic() - audit.started
        dump(report_path, report)
        dump(folder / 'status.json', report)
        return report
    except Exception:
        dump(folder / 'status.json', dict(state='failed', traceback=traceback.format_exc(), historical_mask_fallback=False))
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
