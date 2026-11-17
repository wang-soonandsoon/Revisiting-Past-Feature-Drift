"""CHR paths from cumulative aggregate sufficient statistics.

Only current-task feature rows enter this module. An edge s receives passive
observations from tasks s+1 onward; earlier observations survive as Gram/cross
matrices and counts, never as per-image rows. Maps act on row vectors.
"""

import torch
import torch.nn.functional as F

from ._utils import _cpu64, _fold_metadata, _improves, _paired_inputs


def empty_accumulator():
    """No passive edge exists before task 2."""
    return {'task': 1, 'steps': {}}


def _raw_features(features_by_snapshot, task):
    if task < 1 or any(s not in features_by_snapshot for s in range(task)):
        raise ValueError('Expected snapshots 0 through task-1')
    raw = {s: _cpu64(features_by_snapshot[s]) for s in range(task)}
    first = raw[0]
    if first.ndim != 2 or not first.shape[0] or not first.shape[1]:
        raise ValueError('Expected nonempty raw [n,d] features')
    if any(value.shape != first.shape or not value.isfinite().all()
           for value in raw.values()):
        raise ValueError('All snapshot rows must be finite and aligned')
    return raw


@torch.no_grad()
def update_accumulator(previous, features_by_snapshot, task):
    """Append D_task to each historical edge, returning a new accumulator.

    Edge s maps snapshot s-1 to s. Normalization precedes accumulation; counts
    are actual contributing row counts, not task or class counts.
    """
    if task < 2 or previous['task'] != task - 1:
        raise ValueError('Accumulator must advance by exactly one task from task 1')
    if set(previous['steps']) != set(range(1, task - 1)):
        raise ValueError('Previous accumulator has incomplete historical steps')
    raw = _raw_features(features_by_snapshot, task)
    n, dim = raw[0].shape
    normalized = {s: F.normalize(value, dim=1) for s, value in raw.items()}
    steps = {}
    for s in range(1, task):
        x, y = normalized[s - 1], normalized[s]
        gram, cross, count = x.T @ x, x.T @ y, n
        if s in previous['steps']:
            old = previous['steps'][s]
            old_gram, old_cross = _cpu64(old['gram']), _cpu64(old['cross'])
            if (old_gram.shape != (dim, dim) or old_cross.shape != (dim, dim)
                    or old['count'] <= 0
                    or not old_gram.isfinite().all() or not old_cross.isfinite().all()):
                raise ValueError('Expected finite dimension-matched prior aggregates')
            gram = old_gram + gram
            cross = old_cross + cross
            count += old['count']
        steps[s] = dict(gram=gram, cross=cross, count=count)
    return dict(task=task, steps=steps)


@torch.no_grad()
def select_cumulative_chains(features_by_snapshot, labels, transports, task,
                              alphas, accumulator, *, prior_counts=None):
    """Select one alpha per complete birth path using current whole-class LOCO.

    The accumulator already includes the current task. Each fold subtracts
    only its current held-class contribution; all earlier-task aggregates stay
    intact. Each edge uses lambda=alpha*n_edge_fit/d. The native chain is the
    first candidate; scores use raw held means after the native final W_task.
    Optional prior_counts fixes lambda=alpha*N_prior_edge/d, including in
    held-current-class folds. Omitting it preserves the original recipe.
    """
    if (len(transports) <= task or accumulator['task'] != task
            or set(accumulator['steps']) != set(range(1, task))):
        raise ValueError('Expected current cumulative steps and transports through task')
    raw = _raw_features(features_by_snapshot, task)
    if prior_counts is not None and (set(prior_counts) != set(range(1, task))
                                     or any(n <= 0 for n in prior_counts.values())):
        raise ValueError('Expected a positive fixed prior row count per historical edge')
    _, _, ids, classes, _, bridge = _paired_inputs(
        raw[0], raw[task - 1], labels, transports[task], transports[task])
    n, dim = raw[0].shape
    alphas = tuple(float(alpha) for alpha in alphas)
    if not alphas or any(not 0 < alpha < float('inf') for alpha in alphas):
        raise ValueError('Ridge alphas must be positive and finite')
    native = {s: _cpu64(transports[s]) for s in range(1, task)}
    if any(w.shape != (dim, dim) or not w.isfinite().all() for w in native.values()):
        raise ValueError('Expected finite native passive-step priors')
    totals = {}
    for s in range(1, task):
        entry = accumulator['steps'][s]
        gram, cross, count = _cpu64(entry['gram']), _cpu64(entry['cross']), entry['count']
        if (gram.shape != (dim, dim) or cross.shape != (dim, dim) or count < n
                or not gram.isfinite().all() or not cross.isfinite().all()):
            raise ValueError('Expected finite cumulative aggregates including current rows')
        totals[s] = gram, cross, count

    identity = torch.eye(dim, dtype=torch.float64)
    chains = {}
    for birth in range(task):
        chain = identity
        for s in range(birth + 1, task):
            chain = chain @ native[s]
        chains[birth] = chain
    normalized = {s: F.normalize(value, dim=1) for s, value in raw.items()}
    scores = {b: {'baseline': 0., **{str(a): 0. for a in alphas}} for b in range(task)}
    predictions = {b: {key: [] for key in values} for b, values in scores.items()}
    targets = []
    for label in classes:
        held = ids == label
        held_count = int(held.sum())
        step_solutions = {}
        for s in range(1, task):
            hx, hy = normalized[s - 1][held], normalized[s][held]
            total_gram, total_cross, count = totals[s]
            gram = total_gram - hx.T @ hx
            cross = total_cross - hx.T @ hy
            eig, vectors = torch.linalg.eigh((gram + gram.T) * .5)
            residual = vectors.T @ (cross - gram @ native[s])
            step_solutions[s] = eig.clamp_min(0), vectors, residual, count - held_count
        target = raw[task - 1][held].mean(0) @ bridge
        targets.append(target)
        for birth in range(task):
            mean = raw[birth][held].mean(0)
            baseline = mean @ chains[birth] @ bridge
            predictions[birth]['baseline'].append(baseline)
            scores[birth]['baseline'] += float((baseline - target).square().sum()) / len(classes)
            for alpha in alphas:
                value = mean
                for s in range(birth + 1, task):
                    eig, vectors, residual, n_fit = step_solutions[s]
                    scale_count = n_fit if prior_counts is None else prior_counts[s]
                    value = (value @ native[s]
                             + ((value @ vectors) / (eig + alpha * scale_count / dim)) @ residual)
                prediction = value @ bridge
                predictions[birth][str(alpha)].append(prediction)
                scores[birth][str(alpha)] += float((prediction - target).square().sum()) / len(classes)

    selected = {}
    for birth in range(task):
        best_alpha, best_score = None, scores[birth]['baseline']
        for alpha in alphas:
            if _improves(scores[birth][str(alpha)], best_score):
                best_alpha, best_score = alpha, scores[birth][str(alpha)]
        selected[birth] = best_alpha, best_score

    selected_alphas = {alpha for alpha, _ in selected.values() if alpha is not None}
    steps_by_alpha = {alpha: {} for alpha in selected_alphas}
    for s in range(1, task):
        if not selected_alphas:
            break
        gram, cross, count = totals[s]
        eig, vectors = torch.linalg.eigh((gram + gram.T) * .5)
        eig = eig.clamp_min(0)
        residual = vectors.T @ (cross - gram @ native[s])
        scale_count = count if prior_counts is None else prior_counts[s]
        for alpha in selected_alphas:
            steps_by_alpha[alpha][s] = (native[s]
                + vectors @ (residual / (eig + alpha * scale_count / dim)[:, None]))
    result = {}
    for birth, (alpha, score) in selected.items():
        fitted = chains[birth]
        if alpha is not None:
            fitted = identity
            for s in range(birth + 1, task):
                fitted = fitted @ steps_by_alpha[alpha][s]
        if not fitted.isfinite().all():
            raise ValueError('Nonfinite selected cumulative passive chain')
        result[birth] = dict(map=fitted, alpha=alpha, cv_scores=scores[birth],
            baseline_score=scores[birth]['baseline'], selected_score=score,
            class_count=len(classes), fit_count=n,
            step_fit_counts={s: totals[s][2] for s in range(birth + 1, task)},
            fitting=('cumulative passive normalized ridge, native W_s priors; lambda=alpha*n_step_fit/d'
                     if prior_counts is None else
                     'cumulative passive normalized ridge, native W_s priors; lambda=alpha*N_prior_step/d'),
            selection='one alpha per complete birth path; current whole-class raw LOCO after bridge; '
                      'historical aggregates retained in every fold')
        if prior_counts is not None:
            result[birth]['prior_counts'] = {s: prior_counts[s] for s in range(birth + 1, task)}
        result[birth].update(_fold_metadata(predictions[birth], targets,
            'baseline' if alpha is None else str(alpha), classes))
    return result
