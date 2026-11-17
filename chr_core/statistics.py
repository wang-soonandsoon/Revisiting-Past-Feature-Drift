"""Raw class moments and row-vector linear transport helpers."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassStatistics:
    mean: Tensor
    cov: Tensor
    count: int


Statistics = Mapping[int, ClassStatistics]


@torch.no_grad()
def class_statistics(
    features: Tensor,
    labels: Tensor,
    *,
    ddof: int = 0,
    normalize: bool = False,
    cov_jitter: float = 0.0,
) -> dict[int, ClassStatistics]:
    """Compute birth/current-task statistics, or auditor statistics externally.

    This function has no data-loading capability.  The caller must only give a
    formal learner its current-task features; historical reference features
    belong exclusively to the auditor. Jitter is added once, after estimation.
    """
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Expected features [n,d] and aligned labels [n].")
    if ddof < 0 or cov_jitter < 0:
        raise ValueError("ddof and covariance jitter must be nonnegative.")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf.")
    z = F.normalize(features, dim=-1) if normalize else features
    dtype = z.dtype
    # Full covariance and ill-conditioned ridge geometry need more precision
    # than TF32 provides; keep feature storage and model execution unchanged.
    z = z.double()
    identity = torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    result = {}
    for label in labels.unique(sorted=True):
        rows = z[labels == label]
        count = len(rows)
        if count <= ddof:
            raise ValueError(f"Class {int(label)} has count={count} <= ddof={ddof}.")
        mean = rows.mean(0)
        centered = rows - mean
        covariance = centered.T @ centered / (count - ddof)
        result[int(label)] = ClassStatistics(
            mean.to(dtype).cpu(), (covariance + cov_jitter * identity).to(dtype).cpu(), count
        )
    return result


@torch.no_grad()
def fit_linear_transport(
    before: Tensor,
    after: Tensor,
    *,
    ridge: float,
    mix_alpha: float | None = None,
    normalize: bool = False,
) -> Tensor:
    """Solve ||Y-XW||_F^2 + ridge*||W||_F^2 and optional SLDC mixing.

    Native SLDC uses normalized fitting rows, and mixes the solved W with I
    using omega=exp(-n/(mix_alpha*d)). This differs from ridge centered at I.
    ``normalize=False`` avoids altering an already declared fixed coordinate.
    """
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("Paired features must have equal [n,d] shapes.")
    if ridge < 0 or (mix_alpha is not None and mix_alpha <= 0):
        raise ValueError("ridge must be nonnegative and mix_alpha positive.")
    x = F.normalize(before, dim=-1) if normalize else before
    y = F.normalize(after, dim=-1) if normalize else after
    dtype = x.dtype
    x, y = x.double(), y.double()
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    w = torch.linalg.solve(x.T @ x + ridge * identity, x.T @ y)
    if mix_alpha is not None:
        omega = math.exp(-len(x) / (mix_alpha * x.shape[1]))
        w = (1 - omega) * w + omega * identity
    if not torch.isfinite(w).all():
        raise ValueError("Nonfinite linear transport.")
    return w.to(dtype)


@torch.no_grad()
def transport_statistics(
    statistics: Statistics,
    w: Tensor,
    *,
    bias: Tensor | None = None,
    cov_jitter: float = 0.0,
) -> dict[int, ClassStatistics]:
    """Update aggregate states only: mu'=mu W+b, Sigma'=W.T Sigma W+jI.

    No current or historical per-sample features enter this state update.
    Jitter is an explicit point-estimation choice, not an uncertainty state.
    """
    dtype = w.dtype
    w = w.double()
    identity = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)
    result = {}
    for label, state in statistics.items():
        mean = state.mean.to(w) @ w
        if bias is not None:
            mean = mean + bias
        covariance = w.T @ state.cov.to(w) @ w
        covariance = (covariance + covariance.T) * 0.5
        result[label] = ClassStatistics(
            mean.to(dtype).cpu(), (covariance + cov_jitter * identity).to(dtype).cpu(), state.count
        )
    return result
