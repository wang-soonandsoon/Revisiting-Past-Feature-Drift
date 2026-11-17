"""Mean and Joint CHR corrections with preserved native process noise."""

from dataclasses import replace

import torch

from ._utils import _cpu64


@torch.no_grad()
def corrected_statistics(point, birth, transports, maps, task, cpt,
                         transport_jitter, mode='mean', flip=False):
    """Apply signed direct-map corrections, preserving native no-change states.

    maps[b] is a direct map from birth snapshot b to endpoint task-1. Means use
    mu_P + mu_birth (D-C) W_task, retaining native float32 replay exactly when
    no intervention is selected. Joint covariance uses D W_task and the full
    original accumulated native-jitter term, without inverting any chain.
    Direction mode retains each P norm and the corrected mean's orientation.
    Missing maps retain P. Current classes and counts are never modified.
    flip=True reverses D-C as a signed-direction control, including joint maps.
    """
    if (mode not in ('mean', 'joint', 'direction') or cpt <= 0 or task < 0 or transport_jitter < 0
            or set(point) != set(range((task + 1) * cpt)) or len(transports) <= task
            or any(not 0 <= b < task for b in maps)):
        raise ValueError('Expected complete class states, valid task/maps, mode and nonnegative jitter')
    result = dict(point)
    if not maps:
        return result
    dim = point[0].mean.numel()
    matrices = [None] + [_cpu64(w) for w in transports[1:task + 1]]
    if any(w.shape != (dim, dim) or not w.isfinite().all() for w in matrices[1:]):
        raise ValueError('Expected finite dimension-matched native transports')
    bridge = matrices[task]
    identity = torch.eye(dim, dtype=torch.float64)
    jitter_terms = {}
    if mode == 'joint':
        tail, noise = identity, torch.zeros_like(identity)
        for b in range(task - 1, -1, -1):
            noise = noise + transport_jitter * (tail.T @ tail)
            jitter_terms[b] = noise
            tail = matrices[b + 1] @ tail
    for b, supplied in maps.items():
        chain = identity
        for w in matrices[b + 1:task]:
            chain = chain @ w
        direct = _cpu64(supplied)
        if direct.shape != (dim, dim) or not direct.isfinite().all():
            raise ValueError('Expected finite dimension-matched direct map')
        if torch.equal(direct, chain):
            continue
        delta = direct - chain
        if flip:
            delta = -delta
        full = (chain + delta) @ bridge
        for c in range(b * cpt, (b + 1) * cpt):
            old, initial = point[c], birth[c]
            mean = _cpu64(old.mean) + _cpu64(initial.mean) @ delta @ bridge
            if mode == 'direction':
                norm = mean.norm()
                if norm <= 0:
                    raise ValueError('Direction-only correction requires a nonzero corrected mean')
                mean = mean * (_cpu64(old.mean).norm() / norm)
            covariance = old.cov
            if mode == 'joint':
                covariance = full.T @ _cpu64(initial.cov) @ full + jitter_terms[b]
                covariance = ((covariance + covariance.T) * .5).to(old.cov)
            if not mean.isfinite().all() or not covariance.isfinite().all():
                raise ValueError('Nonfinite corrected statistics')
            result[c] = replace(old, mean=mean.to(old.mean), cov=covariance)
    return result
