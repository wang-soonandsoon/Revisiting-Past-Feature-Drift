"""Shared tensor checks and whole-class selection metadata."""

import torch
import torch.nn.functional as F


def _cpu64(value):
    return value.detach().to(device='cpu', dtype=torch.float64)


def _paired_inputs(before_raw, endpoint_raw, labels, chain, bridge):
    if (before_raw.ndim != 2 or before_raw.shape != endpoint_raw.shape
            or labels.shape != before_raw.shape[:1] or not len(labels)):
        raise ValueError('Expected aligned nonempty raw [n,d] pairs and labels')
    x, y, c, w = map(_cpu64, (before_raw, endpoint_raw, chain, bridge))
    ids = labels.detach().cpu()
    classes = ids.unique(sorted=True)
    dim = x.shape[1]
    if (len(classes) < 3 or not dim or c.shape != (dim, dim) or w.shape != (dim, dim)
            or any(not value.isfinite().all() for value in (x, y, c, w))):
        raise ValueError('Expected at least three classes and finite dimension-matched features/maps')
    return x, y, ids, classes, c, w


def _improves(score, best):
    # Numerical ties retain the earlier candidate, particularly unmodified P.
    return score < best - 1e-12 * max(1., abs(best))


def _fold_metadata(predictions, targets, selected_key, classes):
    target = torch.stack(targets)
    packed = {key: torch.stack(rows) for key, rows in predictions.items()}
    cosines = {key: float(F.cosine_similarity(value, target, dim=1).mean())
               for key, value in packed.items()}
    return dict(baseline_cosine=cosines['baseline'], selected_cosine=cosines[selected_key],
                cv_cosines=cosines, selected_fold_residuals=packed[selected_key] - target,
                class_ids=classes.tolist(),
                residual_definition='held-class prediction minus target, after final bridge')
