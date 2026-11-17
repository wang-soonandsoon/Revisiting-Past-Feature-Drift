"""Current-only historical re-estimation, the matched noncumulative control."""

import torch
import torch.nn.functional as F

from ._utils import _cpu64, _paired_inputs, _improves, _fold_metadata


@torch.no_grad()
def select_passive_chains(features_by_snapshot, labels, transports, task,
                          alphas=(1e-4, 1e-2, 1., 100.)):
    """E1: current-task-only passive one-step refits, then historical chaining.

    Native W_s is each step's ridge prior. One alpha is held fixed for the
    entire path b+1..task-1; it is selected separately for each birth b by
    whole-class LOCO at the final raw held-mean prediction, after W_task.
    The native cumulative chain is an explicit first candidate. This is E1,
    not E1+: no earlier-task image data or extra forwards are consumed.
    """
    if task < 1 or len(transports) <= task or any(b not in features_by_snapshot for b in range(task)):
        raise ValueError('Expected snapshots 0 through task-1 and transports through task')
    raw = {b: _cpu64(features_by_snapshot[b]) for b in range(task)}
    _, _, ids, classes, _, bridge = _paired_inputs(raw[0], raw[task - 1], labels,
                                                  transports[task], transports[task])
    n, dim = raw[0].shape
    if any(value.shape != (n, dim) or not value.isfinite().all() for value in raw.values()):
        raise ValueError('All snapshot rows must be finite and aligned')
    alphas = tuple(float(alpha) for alpha in alphas)
    if not alphas or any(not 0 < alpha < float('inf') for alpha in alphas):
        raise ValueError('Ridge alphas must be positive and finite')
    native = {s: _cpu64(transports[s]) for s in range(1, task)}
    if any(w.shape != (dim, dim) or not w.isfinite().all() for w in native.values()):
        raise ValueError('Expected finite native passive-step priors')
    identity = torch.eye(dim, dtype=torch.float64)
    chains = {}
    for b in range(task):
        chain = identity
        for step in range(b + 1, task):
            chain = chain @ native[step]
        chains[b] = chain
    normalized = {b: F.normalize(value, dim=1) for b, value in raw.items()}
    totals = {s: (normalized[s - 1].T @ normalized[s - 1],
                  normalized[s - 1].T @ normalized[s]) for s in range(1, task)}
    scores = {b: {'baseline': 0., **{str(alpha): 0. for alpha in alphas}}
              for b in range(task)}
    predictions = {b: {key: [] for key in values} for b, values in scores.items()}
    targets = []
    for label in classes:
        held = ids == label
        n_fit = n - int(held.sum())
        # Shared across all births and alphas within this held-class fold.
        step_solutions = {}
        for step in range(1, task):
            hx, hy = normalized[step - 1][held], normalized[step][held]
            gram = totals[step][0] - hx.T @ hx
            cross = totals[step][1] - hx.T @ hy
            eig, vectors = torch.linalg.eigh((gram + gram.T) * .5)
            residual = vectors.T @ (cross - gram @ native[step])
            step_solutions[step] = eig.clamp_min(0), vectors, residual
        target = raw[task - 1][held].mean(0) @ bridge
        targets.append(target)
        for b in range(task):
            mean = raw[b][held].mean(0)
            baseline = mean @ chains[b] @ bridge
            predictions[b]['baseline'].append(baseline)
            scores[b]['baseline'] += float((baseline - target).square().sum()) / len(classes)
            for alpha in alphas:
                value = mean
                for step in range(b + 1, task):
                    eig, vectors, residual = step_solutions[step]
                    value = (value @ native[step]
                             + ((value @ vectors) / (eig + alpha * n_fit / dim)) @ residual)
                prediction = value @ bridge
                predictions[b][str(alpha)].append(prediction)
                scores[b][str(alpha)] += float((prediction - target).square().sum()) / len(classes)

    selected = {}
    for b in range(task):
        best_alpha, best_score = None, scores[b]['baseline']
        for alpha in alphas:
            if _improves(scores[b][str(alpha)], best_score):
                best_alpha, best_score = alpha, scores[b][str(alpha)]
        selected[b] = best_alpha, best_score

    # Refit each required step/alpha once, sharing it between birth paths.
    selected_alphas = {alpha for alpha, _ in selected.values() if alpha is not None}
    steps_by_alpha = {alpha: {} for alpha in selected_alphas}
    for step in range(1, task):
        if not selected_alphas:
            break
        gram, cross = totals[step]
        eig, vectors = torch.linalg.eigh((gram + gram.T) * .5)
        eig = eig.clamp_min(0)
        residual = vectors.T @ (cross - gram @ native[step])
        for alpha in selected_alphas:
            steps_by_alpha[alpha][step] = (native[step]
                + vectors @ (residual / (eig + alpha * n / dim)[:, None]))
    result = {}
    for b, (alpha, score) in selected.items():
        fitted = chains[b]
        if alpha is not None:
            fitted = identity
            for step in range(b + 1, task):
                fitted = fitted @ steps_by_alpha[alpha][step]
        if not fitted.isfinite().all():
            raise ValueError('Nonfinite selected passive chain')
        result[b] = dict(map=fitted, alpha=alpha, cv_scores=scores[b],
                         baseline_score=scores[b]['baseline'], selected_score=score,
                         class_count=len(classes), fit_count=n,
                         fitting='current-task passive one-step normalized ridge, native W_s priors',
                         selection='one alpha per complete birth path; whole-class raw LOCO after bridge')
        result[b].update(_fold_metadata(predictions[b], targets,
                                        'baseline' if alpha is None else str(alpha), classes))
    return result
