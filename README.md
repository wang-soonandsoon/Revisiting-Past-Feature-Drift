# Revisiting Past Feature Drift for Exemplar-Free Class-Incremental Learning

Core implementation of **Cumulative Historical Re-estimation (CHR)**.

CHR revisits the feature maps used to transport old-class statistics. Current
training images pass through saved historical encoders. Their paired features
contribute cumulative correlations, which revise historical maps around the
original estimates. The revised paths update old-class means and, for Joint
CHR, the covariance component transported from class birth.

本仓库整理了论文基础原理对应的核心代码：累计充分统计量、原生映射先验下的岭重估、
整类留出选路，以及 Mean / Joint CHR 的均值与散布修订。示例使用合成特征，可在 CPU
上独立运行。完整图像训练与基准实验编排仍在研究工作区。

## Quick start

Tested with Python 3.11 and PyTorch 2.8.0. From this repository's root:

```bash
python -m pip install -e .
python -m examples.synthetic_demo
python -m unittest discover -s tests -v
```

The example runs five small synthetic tasks. It prints the number of retained
observations, selected revised paths, and illustrative mean errors. These are
synthetic diagnostics, not dataset accuracy or paper benchmark results.

## Code map

| File | Role |
| --- | --- |
| [`chr_core/cumulative.py`](chr_core/cumulative.py) | Accumulate normalized Gram/cross matrices and select/refit complete historical paths |
| [`chr_core/correction.py`](chr_core/correction.py) | Mean and Joint CHR, including the complete original process-noise contribution |
| [`chr_core/current_only.py`](chr_core/current_only.py) | Current-only control with the same path-selection rule |
| [`chr_core/statistics.py`](chr_core/statistics.py) | Class moments and row-vector linear transport helpers |
| [`examples/synthetic_demo.py`](examples/synthetic_demo.py) | A complete feature-level incremental sequence |
| [`tests/`](tests/) | Independent ridge solves, class-held-out checks, prior retention, and noise propagation |

The core functions are extracted from the research implementation. See
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) for their original locations and the
small packaging adaptations.

## How the update works

1. Encode the **same current training image view** at each saved endpoint.
2. Accumulate `G += X.T @ X`, `H += X.T @ Y`, and the contributing image count.
   Each paired feature row is L2 normalized before accumulation.
3. Re-estimate each historical edge using the original map as its ridge prior.
4. Select one regularization value per complete birth path by holding out one
   entire current class at a time. Earlier aggregate observations stay in each
   fold. The original path is also a candidate and wins numerical ties.
5. Refit with all allowed observations, then revise stored class statistics.
   The newest native map is applied exactly once and is shared across variants.

For an edge with original map `W`, the solution is

```python
lam = alpha * observation_count / feature_dimension
A = W + torch.linalg.solve(G + lam * I, H - G @ W)
```

`select_cumulative_chains` uses shared eigendecompositions to evaluate the
same solution efficiently across paths and held-class folds. The default
candidate grid is `(1e-4, 1e-2, 1.0, 100.0)`.

For a class born at task `b`, let `C` and `D` be the original and selected
historical paths ending at snapshot `t-1`, and let `W_t` be the latest map:

```python
mu_chr = mu_native + mu_birth @ (D - C) @ W_t
```

Mean CHR retains the saved native covariance. Joint CHR uses

```python
full_path = D @ W_t
cov_joint = full_path.T @ cov_birth @ full_path + native_accumulated_noise
```

The original accumulated noise includes every original transport step,
propagated through its later native maps. It is not replaced by a single
diagonal jitter. [`docs/METHOD.md`](docs/METHOD.md) gives the equations and
indexing in detail.

## Integration with an incremental learner

Initialize the aggregate state once, before the first revisable task:

```python
from chr_core import (
    DEFAULT_ALPHAS,
    empty_accumulator,
    update_accumulator,
    select_cumulative_chains,
    corrected_statistics,
)

accumulator = empty_accumulator()  # empty after tasks 0 and 1
```

At each task `t >= 2`, after the native learner has produced its new map and
class moments:

```python
# features[s]: aligned raw current TRAIN rows under saved encoder f_s.
# Required keys: 0, ..., t-1. labels identifies each current row's class.
accumulator = update_accumulator(accumulator, features, task=t)
selection = select_cumulative_chains(
    features, labels, native_maps, t, DEFAULT_ALPHAS, accumulator
)
paths = {birth: entry["map"] for birth, entry in selection.items()}

mean_state = corrected_statistics(
    native_state, birth_state, native_maps, paths,
    task=t, cpt=classes_per_task, transport_jitter=0.01, mode="mean",
)
joint_state = corrected_statistics(
    native_state, birth_state, native_maps, paths,
    task=t, cpt=classes_per_task, transport_jitter=0.01, mode="joint",
)
```

Use the revised Gaussians to align a **copy** of the pre-alignment classifier.
The canonical learner and its native state continue the next encoder update.
At inference, use the current encoder and the aligned classifier. Full encoder
training, Gaussian classifier-alignment runners, dataset splits, trained
weights, and benchmark evaluation are outside this core release.

### Input and state conventions

- Tasks and encoder snapshots are zero indexed. Edge `j` maps snapshot `j-1`
  to `j`. `native_maps[0]` is an unused identity placeholder.
- At task `t`, CHR revises edges `1, ..., t-1`. `native_maps[t]` is the shared
  newest bridge. Tasks 0 and 1 retain their native state.
- Rows are feature vectors, and maps multiply on the right. Endpoint tensors
  must have identical row order, augmentation/view, and feature dimension.
  Put historical encoders in evaluation mode for measurement.
- The cumulative state advances once per task. It retains only two matrices
  and a count per edge; it does not retain image rows. Current feature rows are
  temporary inputs to the held-class selection and refit.
- Selection requires at least three current classes. Class labels for selection
  can be arbitrary. `corrected_statistics` follows the original fixed-task-size
  protocol: `cpt` classes per task, with contiguous labels and birth task
  `class_id // cpt`.
- `birth_state` stores the original moments at class arrival. `native_state`
  is the current native linear-compensation state, including current classes.
  Do not feed a previously corrected CHR state back as the native baseline.
- `transport_jitter` must match the native covariance recursion. The main
  recipe adds `0.01 * I` per native step. Sampling-only jitter is separate.
- Fitting uses normalized individual rows; validation, means, and covariance
  transport use raw feature coordinates. Never normalize aggregated means
  or Gaussian samples as part of these updates.
- Held-out test data and old-class reference/Oracle data are not inputs to
  fitting or selection. The synthetic example computes references only after
  fitting to illustrate the update.

The implementation retains several original research controls (`prior_counts`,
`flip`, and `mode="direction"`). The paper's Mean/Joint recipe uses the defaults
and `mode="mean"` or `mode="joint"`.
