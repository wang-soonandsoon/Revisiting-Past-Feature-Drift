"""CPU synthetic checks of cumulative passive ridge and current-class LOCO."""

import copy
import unittest

import torch
import torch.nn.functional as F

from chr_core.current_only import select_passive_chains
from chr_core.cumulative import (empty_accumulator, update_accumulator,
                                     select_cumulative_chains)


DT = torch.float64


def synthetic_sequence():
    generator = torch.Generator().manual_seed(1503)
    a = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=DT)
    b = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=DT)
    c = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=DT)
    actual = [None, a, b, c]
    observations = {}
    labels = {}
    for task, counts in [(2, [2, 4, 5]), (3, [4, 3, 6]), (4, [5, 6, 4])]:
        x = torch.randn(sum(counts), 3, generator=generator, dtype=DT)
        x = x * torch.linspace(.4, 2.7, len(x), dtype=DT)[:, None]
        features = {0: x}
        for step in range(1, task):
            features[step] = features[step - 1] @ actual[step]
        observations[task] = features
        labels[task] = torch.tensor([task * 10 + i for i, n in enumerate(counts)
                                    for _ in range(n)])
    native = [torch.eye(3, dtype=DT),
              torch.tensor([[.6, .1, 0.], [-.1, .7, .2], [.1, 0., .5]], dtype=DT),
              torch.tensor([[.7, 0., -.1], [.2, .6, 0.], [0., .2, .8]], dtype=DT),
              torch.tensor([[.5, -.1, .2], [0., .8, .1], [.1, 0., .7]], dtype=DT),
              torch.tensor([[1.1, .4, -.1], [.2, .8, .3], [0., -.2, 1.2]], dtype=DT)]
    return observations, labels, native


def concatenated_map(observations, current_task, step, prior, alpha,
                     current_mask=None):
    """Independent reference: concatenate rows, then use solve, not eigh."""
    xs, ys = [], []
    for donor_task in range(step + 1, current_task + 1):
        donor = observations[donor_task]
        x, y = donor[step - 1], donor[step]
        if donor_task == current_task and current_mask is not None:
            x, y = x[current_mask], y[current_mask]
        xs.append(F.normalize(x, dim=1))
        ys.append(F.normalize(y, dim=1))
    x, y = torch.cat(xs), torch.cat(ys)
    lam = alpha * len(x) / x.shape[1]
    return torch.linalg.solve(x.T @ x + lam * torch.eye(x.shape[1], dtype=DT),
                              x.T @ y + lam * prior)


class CumulativeGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.observations, self.labels, self.native = synthetic_sequence()
        self.alphas = [.01, 1., 100.]

    def accumulate(self, task):
        accumulator = empty_accumulator()
        for donor in range(2, task + 1):
            accumulator = update_accumulator(accumulator, self.observations[donor], donor)
        return accumulator

    def test_update_matches_concatenated_normalized_rows_without_mutating_previous(self):
        initial = empty_accumulator()
        self.assertEqual(initial, {'task': 1, 'steps': {}})
        second = update_accumulator(initial, self.observations[2], 2)
        saved = copy.deepcopy(second)
        third = update_accumulator(second, self.observations[3], 3)
        self.assertEqual(initial, {'task': 1, 'steps': {}})
        self.assertEqual(second['task'], 2)
        for key in ['gram', 'cross']:
            self.assertTrue(torch.equal(second['steps'][1][key], saved['steps'][1][key]))
        self.assertEqual(third['task'], 3)
        self.assertEqual(set(third['steps']), {1, 2})
        for step in [1, 2]:
            x = torch.cat([F.normalize(self.observations[t][step - 1], dim=1)
                           for t in range(step + 1, 4)])
            y = torch.cat([F.normalize(self.observations[t][step], dim=1)
                           for t in range(step + 1, 4)])
            entry = third['steps'][step]
            self.assertEqual(set(entry), {'gram', 'cross', 'count'})
            self.assertEqual(entry['count'], len(x))
            torch.testing.assert_close(entry['gram'], x.T @ x, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(entry['cross'], x.T @ y, atol=1e-12, rtol=1e-12)
            self.assertEqual(entry['gram'].dtype, DT)
            self.assertEqual(entry['gram'].device.type, 'cpu')
        self.assertFalse(torch.allclose(third['steps'][1]['cross'],
                                        third['steps'][1]['cross'].T))

    def test_cumulative_loco_excludes_only_current_held_class_and_refits_all_donors(self):
        task = 4
        features, ids = self.observations[task], self.labels[task]
        classes = ids.unique(sorted=True)
        result = select_cumulative_chains(features, ids, self.native, task,
                                           self.alphas, self.accumulate(task))
        self.assertEqual(set(result), set(range(task)))
        for birth, branch in result.items():
            scores = {'baseline': 0., **{str(a): 0. for a in self.alphas}}
            residuals = {key: [] for key in scores}
            for label in classes:
                held, train = ids == label, ids != label
                for key in scores:
                    chain = torch.eye(3, dtype=DT)
                    for step in range(birth + 1, task):
                        fitted = self.native[step]
                        if key != 'baseline':
                            fitted = concatenated_map(self.observations, task, step,
                                        self.native[step], float(key), train)
                        chain = chain @ fitted
                    residual = ((features[birth][held].mean(0) @ chain
                                  - features[task - 1][held].mean(0)) @ self.native[task])
                    residuals[key].append(residual)
                    scores[key] += float(residual.square().sum()) / len(classes)
            for key, expected in scores.items():
                self.assertAlmostEqual(branch['cv_scores'][key], expected, places=10)
            chosen = min(scores, key=scores.get)
            self.assertEqual(branch['alpha'], None if chosen == 'baseline' else float(chosen))
            expected_map = torch.eye(3, dtype=DT)
            for step in range(birth + 1, task):
                fitted = self.native[step]
                if branch['alpha'] is not None:
                    fitted = concatenated_map(self.observations, task, step,
                                              self.native[step], branch['alpha'])
                expected_map = expected_map @ fitted
            torch.testing.assert_close(branch['map'], expected_map, atol=2e-11, rtol=2e-11)
            torch.testing.assert_close(branch['selected_fold_residuals'],
                                       torch.stack(residuals[chosen]), atol=2e-11, rtol=2e-11)
            self.assertEqual(branch['fit_count'], len(ids))
            self.assertEqual(branch['class_ids'], classes.tolist())
        self.assertIsNotNone(result[0]['alpha'])
        self.assertFalse(torch.allclose(result[0]['map'], result[0]['map'].T))
        # This fixture must distinguish true cumulative fitting from silently using E1.
        current_only = select_passive_chains(features, ids, self.native, task, self.alphas)
        self.assertGreater(abs(result[0]['cv_scores']['1.0']
                               - current_only[0]['cv_scores']['1.0']), 1e-5)

    def test_first_passive_task_matches_existing_E1(self):
        features, ids = self.observations[2], self.labels[2]
        actual = select_cumulative_chains(features, ids, self.native, 2,
                                           self.alphas, self.accumulate(2))
        expected = select_passive_chains(features, ids, self.native, 2, self.alphas)
        for birth in expected:
            self.assertEqual(actual[birth]['alpha'], expected[birth]['alpha'])
            torch.testing.assert_close(actual[birth]['map'], expected[birth]['map'],
                                       atol=1e-12, rtol=1e-12)
            for key, score in expected[birth]['cv_scores'].items():
                self.assertAlmostEqual(actual[birth]['cv_scores'][key], score, places=12)
            torch.testing.assert_close(actual[birth]['selected_fold_residuals'],
                                       expected[birth]['selected_fold_residuals'],
                                       atol=1e-12, rtol=1e-12)

    def test_exact_native_chain_wins_ties_in_row_vector_order(self):
        a = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=DT)
        b = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=DT)
        native = [torch.eye(3, dtype=DT), a, b, self.native[3]]
        result = select_cumulative_chains(self.observations[3], self.labels[3],
                                           native, 3, self.alphas, self.accumulate(3))
        expected = {0: a @ b, 1: b, 2: torch.eye(3, dtype=DT)}
        self.assertFalse(torch.equal(a @ b, b @ a))
        for birth, branch in result.items():
            self.assertIsNone(branch['alpha'])
            self.assertEqual(branch['selected_score'], branch['baseline_score'])
            self.assertTrue(torch.equal(branch['map'], expected[birth]))

    def test_rejects_wrong_task_state_or_unaligned_nonfinite_rows(self):
        with self.assertRaises(ValueError):
            update_accumulator(empty_accumulator(), self.observations[3], 3)
        second = self.accumulate(2)
        with self.assertRaises(ValueError):
            update_accumulator(second, self.observations[2], 2)
        bad = dict(self.observations[3])
        bad[1] = bad[1][:-1]
        with self.assertRaises(ValueError):
            update_accumulator(second, bad, 3)
        bad = {key: value.clone() for key, value in self.observations[3].items()}
        bad[1][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            update_accumulator(second, bad, 3)
        with self.assertRaises(ValueError):
            select_cumulative_chains(self.observations[3], self.labels[3], self.native,
                                      3, self.alphas, second)


if __name__ == '__main__':
    unittest.main()
