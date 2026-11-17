"""Five-task feature-level example. Its printed errors are synthetic diagnostics."""

import torch

from chr_core import (
    DEFAULT_ALPHAS,
    class_statistics,
    corrected_statistics,
    empty_accumulator,
    select_cumulative_chains,
    transport_statistics,
    update_accumulator,
)


def main():
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(1503)
    dtype, dim, cpt, rows_per_class = torch.float64, 3, 3, 8
    identity = torch.eye(dim, dtype=dtype)
    a = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=dtype)
    b = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=dtype)
    c = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=dtype)
    actual = [identity, a, b, c, a @ b]
    # Deliberately imperfect native maps make historical revision observable.
    native = [identity] + [.7 * w + .3 * identity for w in actual[1:]]
    jitter = .01
    accumulator, births, point = empty_accumulator(), {}, {}

    print("Synthetic CHR demo: 5 tasks, 3 new classes/task, 3-dimensional features")
    print("The errors below are illustrative mean errors, not benchmark accuracy.")
    for task in range(5):
        current_transports = native[:task + 1]
        labels = torch.arange(task * cpt, (task + 1) * cpt).repeat_interleave(rows_per_class)
        centers = 2 * torch.randn(cpt, dim, generator=rng, dtype=dtype)
        x = centers.repeat_interleave(rows_per_class, dim=0)
        x = x + .25 * torch.randn(x.shape, generator=rng, dtype=dtype)
        views = {0: x}
        for step in range(1, task + 1):
            views[step] = views[step - 1] @ actual[step]

        if task:
            point = transport_statistics(point, native[task], cov_jitter=jitter)
        current = class_statistics(views[task], labels, ddof=1)
        births.update(current)
        point.update(current)
        maps, selected = {}, {}
        if task >= 2:
            # All endpoints receive the same current rows. Only G, H and n persist.
            historical_views = {s: views[s] for s in range(task)}
            accumulator = update_accumulator(accumulator, historical_views, task)
            selected = select_cumulative_chains(
                historical_views, labels, current_transports, task, DEFAULT_ALPHAS, accumulator
            )
            maps = {birth: entry["map"] for birth, entry in selected.items()}

        mean = corrected_statistics(point, births, current_transports, maps, task, cpt, jitter, mode="mean")
        joint = corrected_statistics(point, births, current_transports, maps, task, cpt, jitter, mode="joint")
        for label in point:
            torch.testing.assert_close(mean[label].mean, joint[label].mean)
        assert all(mean[int(label)] is point[int(label)] for label in labels.unique())

        # Analytic references are evaluated after fitting and never enter CHR.
        native_errors, revised_errors = [], []
        for label in range(task * cpt):
            path = identity
            for step in range(label // cpt + 1, task + 1):
                path = path @ actual[step]
            reference = births[label].mean @ path
            native_errors.append((point[label].mean - reference).norm())
            revised_errors.append((mean[label].mean - reference).norm())
        before = float(torch.stack(native_errors).mean()) if native_errors else 0.
        after = float(torch.stack(revised_errors).mean()) if revised_errors else 0.
        revised = sum(entry["alpha"] is not None for entry in selected.values())
        counts = {edge: entry["count"] for edge, entry in accumulator["steps"].items()}
        print(f"task={task}  revised_paths={revised}  observations={counts}  "
              f"old_mean_error: native={before:.6f}, CHR={after:.6f}")
        # The native state advances independently; CHR is a readout intervention.


if __name__ == "__main__":
    main()
