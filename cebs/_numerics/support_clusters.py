"""CEBS core numerical implementation."""
from __future__ import annotations

import numpy as np

def discover_trajectory_cluster(supports, template_z, effect_gap, z_threshold=3.0, offset_tolerance=1, minimum_nodes=3):
    """Select a coherent (support start, horizon) translation family without masks."""
    supports = np.asarray(supports, np.int64)
    z = np.asarray(template_z, np.float64)
    gap = np.asarray(effect_gap, np.float64)
    if z.shape != gap.shape or z.shape[0] != len(supports):
        raise ValueError('incompatible template arrays')
    candidates = []
    for length in sorted(set(supports[:, 1].tolist())):
        support_ids = np.flatnonzero(supports[:, 1] == length)
        for offset in range(-z.shape[1], z.shape[1]):
            nodes = []
            for support_index in support_ids:
                start = int(supports[support_index, 0])
                horizon_one_based = np.arange(z.shape[1]) + 1
                eligible = np.flatnonzero((np.abs(horizon_one_based - start - offset) <= offset_tolerance) & (z[support_index] >= z_threshold))
                if len(eligible):
                    horizon = int(eligible[np.argmax(gap[support_index, eligible])])
                    nodes.append((int(support_index), horizon))
            if len(nodes) >= minimum_nodes:
                values = np.asarray([gap[s, h] for (s, h) in nodes])
                strength = float(values.sum() / np.sqrt(len(values)))
                candidates.append((strength, int(length), int(offset), nodes))
    if not candidates:
        raise RuntimeError('no significant translation cluster')
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    (strength, length, offset, nodes) = candidates[0]
    return {'strength': strength, 'length': length, 'offset': offset, 'nodes': nodes}

def prune_nodes(nodes, effect_gap, maximum_nodes):
    gap = np.asarray(effect_gap, np.float64)
    ranked = sorted(nodes, key=lambda node: (-gap[node], node[0], node[1]))
    return ranked[:int(maximum_nodes)]

def exact_budget_alarm(score, count):
    score = np.asarray(score, np.float64)
    ids = np.flatnonzero(np.isfinite(score))
    if len(ids) < count:
        raise ValueError('insufficient finite scores')
    order = np.lexsort((ids, -score[ids]))
    alarm = np.zeros(len(score), bool)
    alarm[ids[order[:count]]] = True
    return alarm

def aligned_event_alarm(input_score, supports, nodes, budget, top_k=4, nms_radius=5):
    """Align window i under support start s to shared absolute coordinate e=i+s."""
    score = np.asarray(input_score, np.float64)
    supports = np.asarray(supports, np.int64)
    n = score.shape[1]
    width = n + int(supports[:, 0].max()) + 1
    aligned = np.full((len(nodes), width), -np.inf, np.float64)
    for (row, (support_index, _)) in enumerate(nodes):
        start = int(supports[support_index, 0])
        aligned[row, start:start + n] = score[support_index]
    aligned = np.where(np.isfinite(aligned), aligned, -1000000000000.0)
    k = min(int(top_k), len(nodes))
    event_score = np.partition(aligned, len(nodes) - k, axis=0)[-k:].mean(axis=0)
    blocked = np.zeros(width, bool)
    provisionally_selected = np.zeros(n, bool)
    groups = []
    order = np.lexsort((np.arange(width), -event_score))
    for event_coordinate in order:
        if blocked[event_coordinate]:
            continue
        ids = np.asarray([event_coordinate - int(supports[s, 0]) for (s, _) in nodes])
        ids = ids[(ids >= 0) & (ids < n)]
        if not np.any(~provisionally_selected[ids]):
            continue
        groups.append((int(event_coordinate), ids))
        provisionally_selected[ids] = True
        lo = max(0, event_coordinate - int(nms_radius))
        hi = min(width, event_coordinate + int(nms_radius) + 1)
        blocked[lo:hi] = True
        if provisionally_selected.sum() >= budget:
            break
    alarm = np.zeros(n, bool)
    remaining = int(budget)
    for (_, ids) in groups:
        ids = ids[~alarm[ids]]
        take = ids[:remaining]
        alarm[take] = True
        remaining -= len(take)
        if remaining == 0:
            break
    if remaining:
        fallback = np.max(np.where(np.isfinite(score), score, -1000000000000.0), axis=0)
        ids = np.flatnonzero(~alarm)
        order = np.lexsort((ids, -fallback[ids]))
        alarm[ids[order[:remaining]]] = True
    return (alarm, event_score, len(groups))
