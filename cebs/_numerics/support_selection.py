"""CEBS core numerical implementation."""
from __future__ import annotations

def active_nodes(cfg, dual):
    gap = dual['observed_strength'] - dual['null_mean']
    cluster = discover_trajectory_cluster(dual['supports'], dual['template_z'], gap, cfg['minimum_permutation_z'], cfg['offset_cluster_tolerance'], cfg['minimum_cluster_nodes'])
    pairs = prune_nodes(cluster['nodes'], gap, cfg['maximum_nodes'])
    active = [{'support_index': int(si), 'support': dual['supports'][si].tolist(), 'horizon_one_based': int(h + 1), 'effect': float(gap[si, h]), 'z': float(dual['template_z'][si, h])} for (si, h) in pairs]
    return (active, {'strength': cluster['strength'], 'length': cluster['length'], 'offset': cluster['offset']})
