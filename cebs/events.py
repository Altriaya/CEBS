"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np
from scipy.stats import t

def fold_risks(mask, blocks):
    return np.asarray([np.asarray(mask, bool)[blocks == b].mean() for b in np.unique(blocks)], dtype=np.float64)

def risk_charge(mask, base, blocks):
    (risks, baseline) = (fold_risks(mask, blocks), fold_risks(base, blocks))
    if len(risks) < 2:
        raise ValueError('At least two nonempty temporal blocks required')
    critical = float(t.ppf(0.95, len(risks) - 1))

    def bound(values):
        return float(values.mean() + critical * values.std(ddof=1) / np.sqrt(len(values)))
    (ucb, base_ucb) = (bound(risks), bound(baseline))
    return dict(empirical=float(np.mean(mask)), fold_risks=risks.tolist(), heuristic_ucb95=ucb, base_heuristic_ucb95=base_ucb, charged=float(np.mean(mask) + max(0.0, ucb - base_ucb)))

def event_length_bins(proposal, blocks, starts):
    """Dyadic length groups fixed independently of dataset and attack.

    bin 0: length 1; bin 1: 2--3; bin 2: 4--7; etc.
    Adjacent rows in an array are not necessarily adjacent physical starts.
    """
    result = np.full(len(proposal), -1, dtype=np.int64)
    for b in np.unique(blocks):
        ids = np.flatnonzero((blocks == b) & proposal)
        ids = ids[np.argsort(starts[ids], kind='stable')]
        if len(ids) == 0:
            continue
        splits = np.flatnonzero(np.diff(starts[ids]) != 1) + 1
        for event in np.split(ids, splits):
            result[event] = int(len(event).bit_length() - 1)
    return result

def supported_bins(bins, proposal, blocks, min_blocks):
    return {int(v): int(len(np.unique(blocks[proposal & (bins == v)]))) for v in np.unique(bins[proposal]) if v >= 0 and len(np.unique(blocks[proposal & (bins == v)])) >= min_blocks}

def select_cut(score, base, proposal, blocks, alpha):
    for cut in np.unique(score[proposal & np.isfinite(score) & (score > 0)]):
        mask = base | proposal & (score >= cut)
        report = risk_charge(mask, base, blocks)
        if report['charged'] <= alpha:
            return (float(cut), mask, report)
    return (None, base.copy(), risk_charge(base, base, blocks))

def calibrate_increment(trusted_score, trusted_base, trusted_proposal, trusted_blocks, trusted_starts, train_score, train_base, train_proposal, train_blocks, train_starts, alpha=0.05, min_support_blocks=3):
    (trusted_score, train_score) = (np.asarray(trusted_score), np.asarray(train_score))
    (trusted_base, train_base) = (np.asarray(trusted_base, bool), np.asarray(train_base, bool))
    (trusted_proposal, train_proposal) = (np.asarray(trusted_proposal, bool), np.asarray(train_proposal, bool))
    (trusted_blocks, train_blocks) = (np.asarray(trusted_blocks), np.asarray(train_blocks))
    (trusted_starts, train_starts) = (np.asarray(trusted_starts), np.asarray(train_starts))
    for arrays in ((trusted_score, trusted_base, trusted_proposal, trusted_blocks, trusted_starts), (train_score, train_base, train_proposal, train_blocks, train_starts)):
        if any((a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays)):
            raise ValueError('All score/mask/reference arrays must have matching one-dimensional rows')
        if len(np.unique(arrays[-1])) != len(arrays[-1]):
            raise ValueError('Duplicate physical window starts')
        if not np.isfinite(arrays[0][arrays[2]]).all():
            raise ValueError('Proposal scores must be finite')
    if len(np.unique(trusted_blocks)) != 12:
        raise ValueError('The frozen protocol requires twelve trusted blocks')
    if np.any(trusted_base & trusted_proposal) or np.any(train_base & train_proposal):
        raise ValueError('Increment proposals must exclude protected base windows')
    if not 0 < alpha < 1 or min_support_blocks < 1:
        raise ValueError('Invalid risk/support settings')
    tbins = event_length_bins(trusted_proposal, trusted_blocks, trusted_starts)
    pbins = event_length_bins(train_proposal, train_blocks, train_starts)
    support = supported_bins(tbins, trusted_proposal, trusted_blocks, min_support_blocks)
    tp = trusted_proposal & np.isin(tbins, list(support))
    pp = train_proposal & np.isin(pbins, list(support))
    (cut, fitted, fitted_risk) = select_cut(trusted_score, trusted_base, tp, trusted_blocks, alpha)
    heldout = trusted_base.copy()
    folds = []
    for b in np.unique(trusted_blocks):
        (fit, test) = (trusted_blocks != b, trusted_blocks == b)
        fit_support = supported_bins(tbins[fit], trusted_proposal[fit], trusted_blocks[fit], min_support_blocks)
        fit_proposal = trusted_proposal[fit] & np.isin(tbins[fit], list(fit_support))
        (c, _, fit_risk) = select_cut(trusted_score[fit], trusted_base[fit], fit_proposal, trusted_blocks[fit], alpha)
        if c is not None:
            heldout[test] |= trusted_proposal[test] & np.isin(tbins[test], list(fit_support)) & (trusted_score[test] >= c)
        folds.append(dict(held_block=int(b), threshold=c, supported_bins=fit_support, fit_charged_risk=fit_risk['charged'], heldout_risk=float(heldout[test].mean()), heldout_additions=int((heldout[test] & ~trusted_base[test]).sum())))
    heldout_risk = risk_charge(heldout, trusted_base, trusted_blocks)
    reasons = []
    if not support:
        reasons.append('insufficient_cross_block_event_length_support')
    if cut is None:
        reasons.append('no_feasible_supported_threshold')
    if fitted_risk['charged'] > alpha:
        reasons.append('fitted_total_risk_exceeds_budget')
    if heldout_risk['charged'] > alpha:
        reasons.append('heldout_threshold_risk_exceeds_budget')
    activate = not reasons
    added = pp & (train_score >= cut) if activate else np.zeros(len(train_base), bool)
    routed = train_base | added
    assert not np.any(train_base & ~routed)
    report = dict(decision='activate_supported_heldout_event' if activate else 'fallback_aligned_base', reasons=reasons, cut=cut, support_bins=support, folds=folds, fitted_risk=fitted_risk, heldout_threshold_risk=heldout_risk, eligible_train_candidates=int(pp.sum()), unsupported_train_candidates=int((train_proposal & ~pp).sum()), added_windows=int(added.sum()), upstream_fully_nested=False, formal_risk_guarantee=False)
    return (dict(routed_hard=routed, added=added, trusted_threshold_oof=heldout, trusted_supported_fitted=fitted, train_supported=pp, trusted_length_bin=tbins, train_length_bin=pbins), report)
