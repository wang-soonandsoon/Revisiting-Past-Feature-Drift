"""Native no-aux SLDC moments, weak propagation and cached-sample CA.

All statistics are raw fp32, as in the upstream implementation.  The RNG scopes
make extra method branches observational: none can change backbone training.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from chr_core import ClassStatistics


@lru_cache(maxsize=1)
def _upstream():
    source = Path(__file__).resolve().parents[1] / 'external/SLDC/models/sldc_modules.py'
    spec = importlib.util.spec_from_file_location('_chr_sldc_modules', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def isolated_rng(seed, device='cpu'):
    """Restore Python, NumPy, CPU and the selected CUDA RNG after a phase."""
    target = torch.device(device)
    devices = ([target.index if target.index is not None else torch.cuda.current_device()]
               if target.type == 'cuda' else [])
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(int(seed))
            np.random.seed(int(seed) % 2**32)
            torch.random.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def native_precision():
    """Use fp32 native statistics/weak/CA math, restoring backbone TF32 policy."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def cached_noise(dim, seed, rows=50000):
    """One shared CPU cache, replacing the upstream unversioned global file."""
    with isolated_rng(seed, 'cpu'):
        return torch.randn(rows, dim)


def _finish(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
@native_precision()
def native_birth(features, labels):
    """Exactly upstream unbiased birth moments; no normalized features/jitter."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError('Expected aligned native features [n,d] and labels [n]')
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError('Native features must be finite raw float32')
    result = {}
    for label in torch.unique(labels):
        rows = features[labels == label]
        if len(rows) < 2:
            raise ValueError('Unbiased native covariance requires at least two rows')
        mean = rows.mean(dim=0)
        centered = rows - mean
        cov = (centered.T @ centered) / (len(rows) - 1)
        result[int(label)] = ClassStatistics(mean.cpu(), cov.cpu(), len(rows))
    return result


@torch.no_grad()
@native_precision()
def native_factor(stat, device='cpu'):
    """Upstream manual factor of covariance + .001 I (sampling only)."""
    cov = stat.cov.to(device)
    return _upstream().cholesky_manual_parallel(
        cov + 1e-3 * torch.eye(cov.size(0), device=cov.device)).cpu()


@torch.no_grad()
@native_precision()
def native_factors(statistics, device='cpu'):
    """Batch the upstream manual recursion across classes, retaining its math.

    This is an execution adaptation: ``bmm`` can round differently from scalar
    ``mm``. The diagonal clamp, sampling jitter and column recursion are the
    source algorithm, with no decomposition replacement or fallback.
    """
    if not statistics:
        return {}
    labels = list(statistics)
    matrix = torch.stack([statistics[c].cov for c in labels]).to(device)
    dim = matrix.shape[-1]
    matrix = matrix + .001*torch.eye(dim, device=device)
    factors = torch.zeros_like(matrix)
    for j in range(dim):
        s_diag = torch.sum(factors[:, j, :j]**2, dim=-1)
        factors[:, j, j] = torch.sqrt(torch.clamp(matrix[:, j, j]-s_diag, min=1e-8))
        if j < dim-1:
            s_off = torch.bmm(factors[:, j+1:, :j], factors[:, j, :j].unsqueeze(-1)).squeeze(-1)
            factors[:, j+1:, j] = (matrix[:, j+1:, j]-s_off)/factors[:, j, j].unsqueeze(-1)
    factors = factors.cpu()
    if not torch.isfinite(factors).all():
        raise ValueError('Nonfinite native sampling factors')
    return {label: factors[index] for index, label in enumerate(labels)}


@torch.no_grad()
@native_precision()
def linear_update(previous, before, after, device, *, gamma_1=1e-4, alpha_t=1.):
    """Native normalized ridge/mixing; transport RAW moments with +.01 I."""
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError('Native linear transport needs aligned [n,d] endpoints')
    started = perf_counter()
    before, after = before.to(device), after.to(device)
    x, y = F.normalize(before, dim=1), F.normalize(after, dim=1)
    identity = torch.eye(x.size(1), device=device)
    w = torch.linalg.solve(x.T @ x + gamma_1 * identity, x.T @ y)
    weight = math.exp(-len(x) / (alpha_t * x.size(1)))
    w = (1 - weight) * w + weight * identity
    if not torch.isfinite(w).all():
        raise ValueError('Nonfinite native linear map')
    result = {}
    for label, old in previous.items():
        mean, cov = old.mean.to(device), old.cov.to(device)
        result[label] = ClassStatistics(
            (mean @ w).cpu(), (w.T @ cov @ w + 1e-2 * identity).cpu(), old.count)
    _finish(device)
    return result, w.cpu(), dict(training_rows=len(x), gamma_1=gamma_1,
        alpha_t=alpha_t, identity_mix=weight, seconds=perf_counter()-started,
        covariance_jitter=.01)


def _compensator(device, gamma_2=.5):
    obj = _upstream().SLDC_Drift_Compensator(dict(
        use_linear_compensation=True, use_weak_nonlinear_compensation=True,
        use_mlp_compensation=False, alpha_t=1., gamma_1=1e-4, gamma_2=gamma_2,
        use_auxiliary_data_enhancement=False))
    obj.device = str(device)
    return obj


@native_precision()
def weak_update(previous, before, after, cached_z, seed, device, *, gamma_2=.5, factors=None):
    """Actual upstream 5000-step compensator and d*10 cached-noise MC.

    Optional ``factors`` are explicit class-indexed factors for previous covs.
    """
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError('Native weak transport needs aligned [n,d] endpoints')
    before, after = before.to(device), after.to(device)
    started = perf_counter()
    with isolated_rng(seed, device):
        obj = _compensator(device, gamma_2)
        mapping, alphas = obj.optimize_nonlinear_transform(before, after)
        _finish(device)
        fit_seconds = perf_counter()-started
        mc_started = perf_counter()
        result, mc_rows = {}, 0
        with torch.no_grad():
            for label, old in previous.items():
                mean = old.mean.to(device)
                factor = (native_factor(old, device) if factors is None else factors[label]).to(device)
                count = mean.numel()*10
                samples = _upstream().sample_torch_cholesky(count, mean, factor, cached_z)
                samples = mapping(F.normalize(samples, dim=1))*samples.norm(dim=1, keepdim=True)
                new_mean = samples.mean(dim=0)
                new_cov = torch.cov(samples.T)+1e-3*torch.eye(mean.numel(), device=device)
                if not torch.isfinite(new_cov).all() or not torch.isfinite(new_mean).all():
                    raise ValueError('Nonfinite weak propagated moments')
                result[label] = ClassStatistics(new_mean.cpu(), new_cov.cpu(), old.count)
                mc_rows += count
        _finish(device)
        mc_seconds = perf_counter()-mc_started
        state = {name: value.detach().cpu().clone() for name, value in mapping.state_dict().items()}
    return result, state, dict(steps=5000, gamma_2=gamma_2, alphas=alphas,
        training_rows=len(before), mc_rows=mc_rows, mc_rows_per_class=before.shape[1]*10,
        fit_seconds=fit_seconds, mc_seconds=mc_seconds, seconds=perf_counter()-started,
        covariance_jitter=.001, sampling_jitter=.001,
        new_factor_count=len(previous) if factors is None else 0)


@native_precision()
def fit_native_head(statistics, canonical, cached_z, seed, device, factors=None):
    """Native six-epoch SCE CA, including global cache shuffle/modulo slices.

    Preserve the canonical head's requires_grad policy.  As upstream, every
    epoch permutes the already permuted tensor, rather than indexing its origin.
    ``factors`` may share the exact native covariance factors across mean-only
    branches; their construction is then charged by the caller, once.
    """
    if not statistics:
        raise ValueError('Native CA needs nonempty statistics')
    started = perf_counter()
    with isolated_rng(seed, device):
        head = deepcopy(canonical).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=5e-5)
        labels, distributions = [], []
        for label, stat in statistics.items():
            labels.append(label)
            factor = native_factor(stat, device) if factors is None else factors[label]
            distributions.append((stat.mean.to(device), factor.to(device)))
        head.train()
        noise = cached_z.to(device)
        noise = noise[torch.randperm(noise.size(0), device=device)]
        samples, targets = [], []
        for label, (mean, factor) in zip(labels, distributions):
            start = (label*1024) % len(noise)
            end = start+1024
            z = (noise[start:end] if end <= len(noise)
                 else torch.cat([noise[start:], noise[:end-len(noise)]], dim=0))
            samples.append(mean+torch.mm(z, factor.t()))
            targets.append(torch.full((1024,), label, device=device))
        inputs, targets = torch.cat(samples), torch.cat(targets)
        _finish(device)
        sampling_seconds = perf_counter()-started
        training_started = perf_counter()
        batch_size = 32*len(statistics)//10
        losses = []
        for _ in range(6):
            order = torch.randperm(len(inputs), device=device)
            inputs, targets = inputs[order], targets[order]
            total = 0.
            for start in range(0, len(inputs), batch_size):
                end = min(start+batch_size, len(inputs))
                optimizer.zero_grad()
                loss = _upstream().symmetric_cross_entropy_loss(head(inputs[start:end]), targets[start:end])
                loss.backward()
                optimizer.step()
                total += loss.item()*(end-start)
            losses.append(total/len(inputs))
            scheduler.step()
        _finish(device)
        if not all(math.isfinite(value) for value in losses):
            raise ValueError('Nonfinite native CA loss')
        head_seconds = perf_counter()-training_started
    head.eval()
    return head, dict(losses=losses, epochs=6, lr=5e-4, batch_size=batch_size,
        weight_decay=1e-4, loss='sce', samples_per_class=1024, training_rows=len(inputs),
        sampling_seconds=sampling_seconds, head_seconds=head_seconds,
        seconds=perf_counter()-started, sampling_jitter=.001,
        factor=('upstream cholesky_manual_parallel' if factors is None
                else 'caller-provided native covariance factors'), cache_rows=len(cached_z),
        new_factor_count=len(statistics) if factors is None else 0)
