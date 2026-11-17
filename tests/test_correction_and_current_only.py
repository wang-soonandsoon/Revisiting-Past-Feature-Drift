"""Independent path fitting and covariance/noise checks from the research suite."""

import unittest

import torch
import torch.nn.functional as F

from chr_core.statistics import ClassStatistics, transport_statistics
from chr_core.correction import corrected_statistics
from chr_core.current_only import select_passive_chains

DT = torch.float64


class CorrectedStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.ws = [torch.eye(2, dtype=DT),
                   torch.tensor([[1.2, .4], [0., .7]], dtype=DT),
                   torch.tensor([[.8, 0.], [-.2, 1.3]], dtype=DT),
                   torch.tensor([[1., .1], [.3, .9]], dtype=DT)]
        self.jitter = .17
        self.birth = {c: ClassStatistics(torch.tensor([c + 1., 2. - c]),
                                        torch.tensor([[.9, .2], [.2, .6]]), 30 + c)
                      for c in range(4)}
        self.point = {}
        for c in range(4):
            state = {c: self.birth[c]}
            for w in self.ws[c + 1:]:
                state = transport_statistics(state, w.float(), cov_jitter=self.jitter)
            self.point[c] = state[c]

    def test_no_change_is_exact_native_objects_and_retains_all_jitter(self):
        maps = {0: self.ws[1] @ self.ws[2], 1: self.ws[2]}
        for mode in ['mean', 'joint']:
            result = corrected_statistics(self.point, self.birth, self.ws, maps,
                                          3, 1, self.jitter, mode)
            self.assertTrue(all(result[c] is self.point[c] for c in self.point))

    def test_noncommuting_joint_transform_adds_original_accumulated_jitter(self):
        chain = self.ws[1] @ self.ws[2]
        delta = torch.tensor([[.3, -.2], [.4, .1]], dtype=DT)
        direct = chain + delta
        result = corrected_statistics(self.point, self.birth, self.ws, {0: direct},
                                      3, 1, self.jitter, mode='joint')
        expected_mean = self.point[0].mean.double() + self.birth[0].mean.double() @ delta @ self.ws[3]
        full = direct @ self.ws[3]
        tail = self.ws[2] @ self.ws[3]
        noise = self.jitter * (tail.T @ tail + self.ws[3].T @ self.ws[3] + torch.eye(2, dtype=DT))
        expected_cov = full.T @ self.birth[0].cov.double() @ full + noise
        torch.testing.assert_close(result[0].mean, expected_mean.float())
        torch.testing.assert_close(result[0].cov, expected_cov.float())
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(result[0].cov).min()), 0.)
        self.assertEqual(result[0].count, self.point[0].count)
        self.assertTrue(all(result[c] is self.point[c] for c in [1, 2, 3]))

    def test_mean_mode_preserves_covariance_and_flip_reverses_only_delta(self):
        chain = self.ws[1] @ self.ws[2]
        direct = chain + torch.eye(2, dtype=DT) * .1
        forward = corrected_statistics(self.point, self.birth, self.ws, {0: direct},
                                        3, 1, self.jitter, mode='mean')
        reverse = corrected_statistics(self.point, self.birth, self.ws, {0: direct},
                                        3, 1, self.jitter, mode='mean', flip=True)
        self.assertIs(forward[0].cov, self.point[0].cov)
        self.assertIs(reverse[0].cov, self.point[0].cov)
        torch.testing.assert_close(forward[0].mean + reverse[0].mean, 2 * self.point[0].mean)

    def test_direction_mode_uses_corrected_direction_and_retains_p_norm(self):
        chain = self.ws[1] @ self.ws[2]
        direct = chain + torch.tensor([[.3, -.1], [-.2, .4]], dtype=DT)
        mean = corrected_statistics(self.point, self.birth, self.ws, {0: direct},
                                    3, 1, self.jitter, mode='mean')[0].mean
        direction = corrected_statistics(self.point, self.birth, self.ws, {0: direct},
                                         3, 1, self.jitter, mode='direction')
        expected = F.normalize(mean, dim=0) * self.point[0].mean.norm()
        torch.testing.assert_close(direction[0].mean, expected)
        self.assertIs(direction[0].cov, self.point[0].cov)
        self.assertTrue(all(direction[c] is self.point[c] for c in [1, 2, 3]))
        baseline = corrected_statistics(self.point, self.birth, self.ws, {0: chain},
                                         3, 1, self.jitter, mode='direction')
        self.assertTrue(all(baseline[c] is self.point[c] for c in self.point))


class PassiveChainTests(unittest.TestCase):
    def test_loco_compares_complete_paths_with_one_shared_alpha(self):
        torch.set_num_threads(4)
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(12, 2, generator=generator, dtype=DT)
        ids = torch.tensor([0] * 3 + [1] * 5 + [2] * 4)
        a = torch.tensor([[0., -1.], [1., 0.]], dtype=DT)
        b = torch.tensor([[1., 0.], [0., -1.]], dtype=DT)
        features = {0: x, 1: x @ a, 2: x @ a @ b}
        ws = [torch.eye(2, dtype=DT), torch.eye(2, dtype=DT) * .8,
              torch.tensor([[.7, .1], [-.1, .8]], dtype=DT),
              torch.tensor([[1., .2], [-.3, .9]], dtype=DT)]
        alphas = [.01, 1.]
        actual = select_passive_chains(features, ids, ws, task=3, alphas=alphas)
        self.assertEqual(set(actual), {0, 1, 2})
        for birth in range(3):
            scores = {'baseline': 0., **{str(alpha): 0. for alpha in alphas}}
            fold_residuals = {key: [] for key in scores}
            for held_class in range(3):
                held, train = ids == held_class, ids != held_class
                for key in scores:
                    chain = torch.eye(2, dtype=DT)
                    for step in range(birth + 1, 3):
                        fitted = ws[step]
                        if key != 'baseline':
                            sx, sy = (F.normalize(features[s][train], dim=1)
                                      for s in [step - 1, step])
                            lam = float(key) * int(train.sum()) / 2
                            fitted = torch.linalg.solve(sx.T @ sx + lam * torch.eye(2, dtype=DT),
                                                        sx.T @ sy + lam * ws[step])
                        chain = chain @ fitted
                    residual = ((features[birth][held].mean(0) @ chain
                                 - features[2][held].mean(0)) @ ws[3])
                    fold_residuals[key].append(residual)
                    scores[key] += float(residual.square().sum()) / 3
            for key, expected in scores.items():
                self.assertAlmostEqual(actual[birth]['cv_scores'][key], expected, places=12)
            selected = 'baseline' if actual[birth]['alpha'] is None else str(actual[birth]['alpha'])
            torch.testing.assert_close(actual[birth]['selected_fold_residuals'],
                                       torch.stack(fold_residuals[selected]), atol=3e-13, rtol=0)
            fitted = torch.eye(2, dtype=DT)
            for step in range(birth + 1, 3):
                p = ws[step]
                if selected != 'baseline':
                    sx, sy = (F.normalize(features[s], dim=1) for s in [step - 1, step])
                    lam = float(selected) * len(ids) / 2
                    p = torch.linalg.solve(sx.T @ sx + lam * torch.eye(2, dtype=DT),
                                           sx.T @ sy + lam * ws[step])
                fitted = fitted @ p
            torch.testing.assert_close(actual[birth]['map'], fitted, atol=3e-13, rtol=0)
        self.assertIsNone(actual[2]['alpha'])

    def test_exact_native_passive_paths_keep_original_chain(self):
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(9, 2, generator=generator, dtype=DT)
        ids = torch.arange(9) % 3
        a = torch.tensor([[0., -1.], [1., 0.]], dtype=DT)
        b = torch.tensor([[1., 0.], [0., -1.]], dtype=DT)
        features = {0: x, 1: x @ a, 2: x @ a @ b}
        ws = [torch.eye(2, dtype=DT), a, b, torch.eye(2, dtype=DT)]
        result = select_passive_chains(features, ids, ws, task=3, alphas=[.01, 1.])
        for birth in range(3):
            self.assertIsNone(result[birth]['alpha'])
            self.assertAlmostEqual(result[birth]['baseline_cosine'], 1.)
        self.assertTrue(torch.equal(result[0]['map'], a @ b))


if __name__ == "__main__":
    unittest.main()
