"""CEBS core numerical implementation."""
from __future__ import annotations

def support_bounds(key, normal_step_q95):
    informative = np.flatnonzero(np.abs(np.asarray(key)) * exp.core.BIN_WIDTH > normal_step_q95)
    if not len(informative):
        return (0, len(key) + 1, False)
    return (int(informative[0]), int(informative[-1] + 2), True)

def background_denominator(scores, block_ids):
    """Positive precursor-conditioned event background, not overlapping windows."""
    (scores, block_ids) = (np.asarray(scores), np.asarray(block_ids))
    if len(scores) < 8 or len(np.unique(block_ids)) < 3:
        return (1.0, False)
    return (max(1.0, float(np.quantile(scores, 0.95))), True)

def stamp(scores, begins, channel, lo, hi, strength):
    if not len(begins):
        return
    index = np.asarray(begins, np.int64)[:, None] + np.arange(lo, hi)
    scores[index, channel] = np.maximum(scores[index, channel], strength)

class ControlledVariants:

    def __init__(self, name):
        self.name = name
        self.models = []
        self.components = []
        self.t0 = time.monotonic()

    def mark(self, *args):
        pass

    def selected_hits(self, *args):
        pass

    def on_fold(self, fold, motifs, accepted, data, raws, tfits, thelds, tid, scale, prec, tprec):
        eligible = [m for (m, accept) in zip(motifs, accepted) if accept and m['strength'] > 0 and (len(exp.core.independent_positions(m['hit'])) >= exp.core.MIN_QUERY_EVENTS)]
        groups = defaultdict(list)
        for motif in eligible:
            groups[motif['channel'], motif['length']].append(motif)
        trusted = data['trusted']
        bg_prec = {}
        for b in tfits:
            ids = np.flatnonzero(trusted['block_ids'] == b)
            ref = tid & (trusted['block_ids'] != b)
            bg_prec[int(b)] = exp.core.precursor(exp.core.percent_flags(data['trusted_scores'][:, ref], data['trusted_scores'][:, ids]), data['supports'], len(raws[b]))
        normal_step = np.quantile(np.concatenate([np.abs(np.diff(raws[b], axis=0)) for b in tfits]), 0.95, axis=0) / scale
        for ((channel, length), members) in groups.items():
            keys = exp.row_keys(np.asarray([m['key'] for m in members], np.int32))
            order = np.argsort(keys)
            keys = keys[order]
            strengths = np.asarray([m['strength'] for m in members])[order]
            lags = np.asarray([m['lag'] - 1 for m in members], np.int64)[order]
            (bg_scores, bg_blocks) = ([], [])
            for b in tfits:
                (q, nonflat) = exp.core.motif_rows(raws[b], channel, length, float(scale[channel]))
                (indices, found) = exp.match_groups(keys, exp.row_keys(q))
                starts = np.flatnonzero(found & nonflat)
                if not len(starts):
                    continue
                which = indices[starts]
                starts = starts[bg_prec[int(b)][starts, lags[which]] > 0]
                starts = exp.core.independent_positions(starts)
                bg_scores.extend(strengths[indices[starts]].tolist())
                bg_blocks.extend([int(b)] * len(starts))
            (denominator, reliable) = background_denominator(bg_scores, bg_blocks)
            self.models.append({'fold': fold, 'channel': channel, 'length': length, 'normal_independent_events': len(bg_scores), 'normal_blocks': len(set(bg_blocks)), 'background_q95_denominator': denominator, 'background_reliable': reliable, 'normal_step_q95': float(normal_step[channel])})
            for motif in members:
                (lo, hi, crop_evidence) = support_bounds(motif['key'], normal_step[channel])
                hp = motif['hit'][prec[motif['hit'], motif['lag'] - 1] > 0]
                raw = motif['strength']
                adjusted = raw / denominator
                ranges = {'background_only': (0, length, adjusted), 'support_only': (lo, hi, raw), 'background_support': (lo, hi, adjusted)}
                for (variant, (begin_offset, end_offset, strength)) in ranges.items():
                    stamp(self.point[variant], hp, channel, begin_offset, end_offset, strength)
                    for b in thelds:
                        tp = motif['trusted_hit'][int(b)]
                        tp = tp[tprec[int(b)][tp, motif['lag'] - 1] > 0]
                        stamp(self.tpoint[variant][b], tp, channel, begin_offset, end_offset, strength)
                self.components.append((fold, channel, length, motif['count'], motif['excess'], raw, denominator, adjusted, lo, hi, crop_evidence, len(hp)))
