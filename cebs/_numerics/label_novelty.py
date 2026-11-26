"""CEBS core numerical implementation."""
from __future__ import annotations

def strict_novelty(old_evidence: np.ndarray) -> np.ndarray:
    """No same-channel old evidence within the overlap-coupled radius."""
    expanded = np.zeros_like(old_evidence)
    for delta in range(-RADIUS, RADIUS + 1):
        if delta < 0:
            expanded[:delta] |= old_evidence[-delta:]
        elif delta > 0:
            expanded[delta:] |= old_evidence[:-delta]
        else:
            expanded |= old_evidence
    return ~expanded
