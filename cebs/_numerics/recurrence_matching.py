"""CEBS core numerical implementation."""
from __future__ import annotations

def row_keys(rows):
    rows = np.ascontiguousarray(rows)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()

def match_groups(keys, query):
    where = np.searchsorted(keys, query)
    clipped = np.minimum(where, len(keys) - 1)
    valid = (where < len(keys)) & (keys[clipped] == query)
    return (where, valid)

def corridor_mask(events, cutoff, bounds, n):
    selected = np.flatnonzero(events >= cutoff)
    difference = np.zeros(n + 1, np.int64)
    lo = np.clip(selected + bounds[0], 0, n)
    hi = np.clip(selected + bounds[1] + 1, 0, n)
    keep = lo < hi
    np.add.at(difference, lo[keep], 1)
    np.add.at(difference, hi[keep], -1)
    return np.cumsum(difference[:-1]) > 0

def threshold_boundaries(trusted_scores):
    """Exhaust every >= threshold step; uniform float64 score storage required."""
    values = np.unique(np.concatenate([s[np.isfinite(s) & (s > 0)] for s in trusted_scores]))
    candidates = np.unique(np.r_[np.nextafter(0.0, 1.0), values, np.nextafter(values.astype(np.float64), np.inf), np.inf])
    total = sum((len(s) - 23 for s in trusted_scores))

    def risk(cutoff):
        return sum((int(core.window_mask(s >= cutoff).sum()) for s in trusted_scores)) / total
    (left, right) = (0, len(candidates) - 1)
    while left < right:
        middle = (left + right) // 2
        if risk(candidates[middle]) <= core.TRUSTED_RISK:
            right = middle
        else:
            left = middle + 1
    return (float(candidates[left]), float(risk(candidates[left])))

def freeze(name, observer=None):
    dest = OUT / name.lower()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'frozen_preoracle.npz').exists():
        return
    core.dump(dest / 'protocol.json', {**core.protocol(), 'dataset': name, 'variants': ['recurrence_only', 'paired_closure', 'paired_corridor_same_hits'], 'recurrence_only': 'all eligible fit motifs and held recurrence, no input, no pairing BH; score log1p(fit occurrences); same trusted risk calibration', 'corridor_control': 'same selected motif hits and input event centers begin-lag at paired-closure cutoff; original v2 corridor, no NMS; not separately risk-selected', 'time_blocks': 'largest multiple of four <=12 with >=4 complete dataset phase cycles in every block; trusted blocks remain original12', 'phase': 'existing dataset period: ETT96, PEMS288, Weather144; not attacker metadata', 'numerical_boundary_correction': CORRECTED_BOUNDARIES, 'boundary_fix': 'when enabled: score storage float64 everywhere; exhaustive >= boundary candidates include nextafter observed values; exact monotone binary search, no risk relaxation', 'no_training': True})
    data = prepare(name)
    (train, trusted) = (data['train'], data['trusted'])
    raws = core.reconstruct_trusted(trusted)
    (n, channels) = train.shape
    cycles = n // data['period']
    nblocks = 4 * (min(12, cycles // 4) // 4)
    if nblocks < 4:
        raise ValueError('insufficient phase cycles for legal null and cross-fitting')
    edges = np.linspace(0, cycles, nblocks + 1, dtype=int) * data['period']
    blocks = np.searchsorted(edges[1:], np.arange(n), side='right')
    if observer is not None:
        observer.begin(data, edges)
    point = {v: np.full_like(train, -np.inf) for v in ('recurrence_only', 'paired_closure')}
    tpoint = {v: [np.full(r.shape, -np.inf, dtype=np.float64 if CORRECTED_BOUNDARIES else r.dtype) for r in raws] for v in point}
    events = np.full(n, -np.inf)
    tevents = [np.full(len(r), -np.inf) for r in raws]
    manifest = []
    t0 = time.monotonic()
    for fold in range(4):
        fit_blocks = np.flatnonzero(np.arange(nblocks) % 4 != fold)
        held_blocks = np.flatnonzero(np.arange(nblocks) % 4 == fold)
        tfits = np.flatnonzero(np.arange(12) % 4 != fold)
        thelds = np.flatnonzero(np.arange(12) % 4 == fold)
        tid = np.isin(trusted['block_ids'], tfits)
        scale = np.maximum(np.quantile(np.concatenate([np.abs(np.diff(raws[b], axis=0)) for b in tfits]), 0.75, axis=0), 0.001)
        flags = core.percent_flags(data['trusted_scores'][:, tid], data['poison_scores'])
        prec = core.precursor(flags, data['supports'], n)
        tprec = {}
        for b in thelds:
            ids = np.flatnonzero(trusted['block_ids'] == b)
            tprec[int(b)] = core.precursor(core.percent_flags(data['trusted_scores'][:, tid], data['trusted_scores'][:, ids]), data['supports'], len(raws[b]))
        rng = np.random.default_rng(20260918 + fold)
        shifts = {int(b): rng.integers(2, (edges[b + 1] - edges[b]) // data['period'] - 1, size=core.PERMUTATIONS) * data['period'] for b in fit_blocks}
        motifs = []
        for c in range(channels):
            for length in core.LENGTHS:
                (q, nonflat) = core.motif_rows(train, c, length, float(scale[c]))
                pos = np.arange(len(q))
                valid = (blocks[pos] < nblocks) & (blocks[pos] == blocks[pos + length - 1]) & nonflat
                fit = valid & np.isin(blocks[pos], fit_blocks)
                held = valid & np.isin(blocks[pos], held_blocks)
                (fitpos, heldpos) = (pos[fit], pos[held])
                if observer is not None:
                    observer.mark('nonflat', heldpos, c, length)
                if not len(fitpos):
                    continue
                allkeys = row_keys(q)
                (keys, inv, counts) = np.unique(allkeys[fit], return_inverse=True, return_counts=True)
                order = np.argsort(inv, kind='stable')
                bounds = np.r_[0, np.cumsum(counts)]
                (query_group, hit) = match_groups(keys, allkeys[held])
                qorder = np.argsort(query_group[hit], kind='stable')
                (matchedpos, matchedgroup) = (heldpos[hit][qorder], query_group[hit][qorder])
                if observer is not None:
                    observer.mark('exact_fit_match', matchedpos, c, length)
                query_bounds = np.searchsorted(matchedgroup, np.arange(len(keys) + 1))
                trust_hit = {}
                for b in thelds:
                    (tq, tn) = core.motif_rows(raws[b], c, length, float(scale[c]))
                    (groups, found) = match_groups(keys, row_keys(tq))
                    keep = found & tn
                    (gp, tp) = (groups[keep], np.flatnonzero(keep))
                    to = np.argsort(gp, kind='stable')
                    (gp, tp) = (gp[to], tp[to])
                    trust_hit[int(b)] = (tp, np.searchsorted(gp, np.arange(len(keys) + 1)))
                for group in np.flatnonzero(counts >= core.MIN_FIT_EVENTS):
                    occ = core.independent_positions(np.sort(fitpos[order[bounds[group]:bounds[group + 1]]]))
                    if len(occ) < core.MIN_FIT_EVENTS or len(np.unique(blocks[occ])) < core.MIN_FIT_BLOCKS:
                        continue
                    observed = prec[occ].mean(axis=0)
                    null_sum = np.zeros((core.PERMUTATIONS, 24), np.float64)
                    for b in np.unique(blocks[occ]):
                        ob = occ[blocks[occ] == b]
                        (unique_shift, inverse_shift) = np.unique(shifts[int(b)], return_inverse=True)
                        nullpos = edges[b] + (ob[None, :] - edges[b] + unique_shift[:, None]) % (edges[b + 1] - edges[b])
                        null_sum += prec[nullpos].sum(axis=1)[inverse_shift]
                    null = null_sum / len(occ)
                    lag = int(observed.argmax())
                    p = float((1 + (null.max(axis=1) >= observed[lag]).sum()) / 1000)
                    excess = max(float(observed[lag]) - float(np.quantile(null.max(axis=1), 0.95)), 0.0)
                    hp = matchedpos[query_bounds[group]:query_bounds[group + 1]]
                    if observer is not None:
                        observer.mark('fit_recurrence', hp, c, length)
                    th = {int(b): trust_hit[int(b)][0][trust_hit[int(b)][1][group]:trust_hit[int(b)][1][group + 1]] for b in thelds}
                    motifs.append({'channel': c, 'length': length, 'key': q[fitpos[order[bounds[group]]]].tolist(), 'count': len(occ), 'lag': lag + 1, 'p': p, 'excess': excess, 'strength': float(np.log1p(len(occ)) * excess), 'hit': hp, 'trusted_hit': th})
            if c % 25 == 0 or c == channels - 1:
                print(json.dumps({'dataset': name, 'fold': fold, 'channel': c, 'channels': channels, 'eligible_motifs': len(motifs), 'seconds': round(time.monotonic() - t0, 1)}), flush=True)
        accepted = core.bh(np.asarray([m['p'] for m in motifs]))
        if observer is not None and hasattr(observer, 'on_fold'):
            observer.on_fold(fold, motifs, accepted, data, raws, tfits, thelds, tid, scale, prec, tprec)
        paired_count = recurrence_count = 0
        for (m, accept) in zip(motifs, accepted):
            (hp, c, length) = (m['hit'], m['channel'], m['length'])
            if len(core.independent_positions(hp)) < core.MIN_QUERY_EVENTS:
                continue
            recurrence_count += 1
            if observer is not None:
                observer.mark('held_recurrence', hp, c, length)
            variants = [('recurrence_only', float(np.log1p(m['count'])), False)]
            if accept and m['strength'] > 0:
                variants.append(('paired_closure', m['strength'], True))
                paired_count += 1
                if observer is not None:
                    observer.mark('paired_significance', hp, c, length)
                    observer.mark('input_paired', hp[prec[hp, m['lag'] - 1] > 0], c, length)
                    observer.selected_hits(fold, m, prec)
            for (variant, strength, paired) in variants:
                for begin in hp:
                    if paired and prec[begin, m['lag'] - 1] == 0:
                        continue
                    point[variant][begin:begin + length, c] = np.maximum(point[variant][begin:begin + length, c], strength)
                    if paired:
                        center = begin - m['lag']
                        if center >= 0:
                            events[center] = max(events[center], strength)
                for b in thelds:
                    for begin in m['trusted_hit'][int(b)]:
                        if paired and tprec[int(b)][begin, m['lag'] - 1] == 0:
                            continue
                        tpoint[variant][b][begin:begin + length, c] = np.maximum(tpoint[variant][b][begin:begin + length, c], strength)
                        if paired:
                            center = begin - m['lag']
                            if center >= 0:
                                tevents[b][center] = max(tevents[b][center], strength)
        manifest.append({'fold': fold, 'tested': len(motifs), 'BH_accepted': int(accepted.sum()), 'unique_null_shift_schedules': int(len(np.unique(np.stack([shifts[int(b)] for b in fit_blocks], axis=1), axis=0))), 'held_recurrence': recurrence_count, 'held_paired': paired_count})
        core.dump(dest / 'progress_preoracle.json', manifest)
    saved = {'baseline': data['baseline'], 'supports': data['supports'], 'train_block_edges': edges}
    decisions = {}
    for variant in point:
        (cutoff, risk) = threshold_boundaries(tpoint[variant]) if CORRECTED_BOUNDARIES else core.choose_threshold(tpoint[variant])
        cells = point[variant] >= cutoff
        mask = core.window_mask(cells)
        assert len(mask) == len(data['baseline'])
        saved[variant] = mask
        saved[variant + '_cells'] = cells
        decisions[variant] = {'threshold': cutoff, 'trusted_risk': risk, 'deleted': int(mask.sum()), 'physical_cells': int(cells.sum())}
    paired_cutoff = decisions['paired_closure']['threshold']
    saved['paired_corridor_same_hits'] = corridor_mask(events, paired_cutoff, data['bounds'], n - 23)
    decisions['paired_corridor_same_hits'] = {'threshold': paired_cutoff, 'trusted_risk': sum((int(corridor_mask(e, paired_cutoff, data['bounds'], len(raws[b]) - 23).sum()) for (b, e) in enumerate(tevents))) / len(trusted['x']), 'deleted': int(saved['paired_corridor_same_hits'].sum()), 'risk_selected': False}
    original_hash = hashlib.sha256(data['baseline'].tobytes()).hexdigest()
    np.savez_compressed(dest / 'frozen_preoracle.npz', **saved)
    core.dump(dest / 'preoracle.json', {'dataset': name, 'decisions': decisions, 'folds': manifest, 'original_nodes': data['supports'].tolist(), 'original_corridor': data['bounds'], 'original_budget_windows': len(trusted['x']), 'original_budget_raw_points': sum((len(r) for r in raws)), 'reference_lineage': data['lineage'], 'period': data['period'], 'time_blocks': nblocks, 'baseline_mask_sha256': original_hash, 'source_poison_sha256': hashlib.sha256((ROOT / data['poison_path']).read_bytes()).hexdigest(), 'state': 'frozen_before_oracle', 'elapsed_seconds': time.monotonic() - t0})
    if observer is not None:
        observer.finish(saved, decisions, dest)
