"""Small independent upstream comparisons for the native SLDC operations."""
import copy
import importlib.util
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chr_training.heads import ContinualLinear
from chr_training.native import (
    cached_noise, fit_native_head, isolated_rng, linear_update, native_birth,
    native_factor, native_factors, native_precision, weak_update,
)


def upstream():
    path = Path(__file__).resolve().parents[1] / 'external/SLDC/models/sldc_modules.py'
    spec = importlib.util.spec_from_file_location('_test_sldc_modules', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compensator(module):
    obj = module.SLDC_Drift_Compensator(dict(
        use_linear_compensation=True, use_weak_nonlinear_compensation=True,
        use_mlp_compensation=False, alpha_t=1., gamma_1=1e-4, gamma_2=.5,
        use_auxiliary_data_enhancement=False))
    obj.device = 'cpu'
    return obj


class NativeBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.up = upstream()

    def fixture(self):
        g = torch.Generator().manual_seed(19)
        x = torch.randn(15, 3, generator=g)
        y = x @ torch.tensor([[.9, .1, .0], [.0, 1.1, .1], [.1, .0, .95]]) + .02
        labels = torch.arange(3).repeat_interleave(5)
        return x, y, labels

    def test_birth_and_sampling_factor_match_upstream_raw_float32(self):
        x, _, labels = self.fixture()
        actual = native_birth(x, labels)
        expected = compensator(self.up).compute_class_statistics(x, labels)
        for c in actual:
            self.assertTrue(torch.equal(actual[c].mean, expected[c].mean))
            self.assertTrue(torch.equal(actual[c].cov, expected[c].cov))
            self.assertEqual(actual[c].count, 5)
            self.assertTrue(torch.equal(native_factor(actual[c], 'cpu'), expected[c].L))
        # Covariance is unbiased RAW covariance, not normalized or jittered at birth.
        self.assertTrue(torch.allclose(actual[0].cov, torch.cov(x[:5].T)))

    def test_batched_manual_factors_match_independent_upstream(self):
        x, _, labels = self.fixture()
        statistics = native_birth(x, labels)
        expected = compensator(self.up).compute_class_statistics(x, labels)
        with patch('chr_training.native.native_factor', side_effect=AssertionError('Scalar fallback')):
            actual = native_factors(statistics, 'cpu')
        self.assertEqual(list(actual), list(statistics))
        for c in statistics:
            torch.testing.assert_close(actual[c], expected[c].L, rtol=1e-6, atol=1e-7)

    def test_native_precision_scope_restores_outer_tf32_policy(self):
        outer = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with native_precision():
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                with native_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with native_precision():
                    raise RuntimeError('fixture')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = outer

    def test_linear_transform_matches_upstream_noaux_and_preserves_input(self):
        x, y, labels = self.fixture()
        old = native_birth(x, labels)
        snapshot = copy.deepcopy(old)
        source = {c: self.up.GaussianStatistics(s.mean, s.cov) for c, s in old.items()}
        expected = compensator(self.up).update_statistics_with_linear_transform(source, x, y)
        actual, w, info = linear_update(old, x, y, 'cpu')
        self.assertEqual(info['training_rows'], len(x))
        self.assertEqual(info['gamma_1'], 1e-4)
        self.assertEqual(info['alpha_t'], 1.)
        for c in old:
            torch.testing.assert_close(actual[c].mean, expected[c].mean, rtol=0, atol=0)
            torch.testing.assert_close(actual[c].cov, expected[c].cov, rtol=0, atol=0)
            self.assertEqual(actual[c].count, old[c].count)
            self.assertTrue(torch.equal(old[c].mean, snapshot[c].mean))
            self.assertTrue(torch.equal(old[c].cov, snapshot[c].cov))
            torch.testing.assert_close(actual[c].mean, old[c].mean @ w)

    def test_all_rng_families_preserved_and_reproducible(self):
        random.seed(103); np.random.seed(103); torch.manual_seed(103)
        py, np_state, tc = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        with isolated_rng(7, 'cpu'):
            values = (random.random(), float(np.random.rand()), torch.rand(2))
        self.assertEqual(random.getstate(), py)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_state)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), tc))
        with isolated_rng(7, 'cpu'):
            self.assertEqual(random.random(), values[0]); self.assertEqual(float(np.random.rand()), values[1])
            self.assertTrue(torch.equal(torch.rand(2), values[2]))
        self.assertTrue(torch.equal(cached_noise(3, 7, rows=31), cached_noise(3, 7, rows=31)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), tc))

    def test_full_native_ca_matches_upstream_including_noise_wrap(self):
        x, _, labels = self.fixture()
        stats = native_birth(x, labels)
        with isolated_rng(41, 'cpu'):
            canonical = ContinualLinear(3, 1)
            canonical.update(2)  # Previous head frozen, new head plus scale trainable.
        before = copy.deepcopy(canonical.state_dict())
        z = cached_noise(3, 5, rows=1300)  # class 1/2 slices wrap around cache end.
        obj = compensator(self.up)
        obj.cached_Z = z
        source = {c: self.up.GaussianStatistics(s.mean, s.cov) for c, s in stats.items()}
        with isolated_rng(31, 'cpu'):
            expected = obj.train_classifier_with_cached_samples(copy.deepcopy(canonical), source)
        actual, info = fit_native_head(stats, canonical, z, 31, 'cpu')
        factors = {c: source[c].L for c in stats}
        with patch('chr_training.native.native_factor', side_effect=AssertionError('Recomputed cached factor')):
            cached, cached_info = fit_native_head(stats, canonical, z, 31, 'cpu', factors=factors)
        self.assertEqual(info['epochs'], 6)
        self.assertEqual(info['training_rows'], 3072)
        self.assertEqual(info['batch_size'], 9)
        self.assertEqual(len(info['losses']), 6)
        for key in actual.state_dict():
            torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(cached.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(torch.equal(canonical.state_dict()[key], before[key]))
        self.assertEqual(info['losses'], cached_info['losses'])
        self.assertFalse(actual.heads[0].weight.requires_grad)
        self.assertTrue(actual.heads[1].weight.requires_grad)

    def test_weak_fits_actual_upstream_5000_and_propagates_exact_mc(self):
        x, y, labels = self.fixture()
        old = native_birth(x, labels)
        z = cached_noise(3, 8, rows=1300)
        obj = compensator(self.up)
        obj.cached_Z = z
        source = {c: self.up.GaussianStatistics(s.mean, s.cov) for c, s in old.items()}
        with isolated_rng(53, 'cpu'):
            expected = obj.update_statistics_with_weak_nonlinear_transform(source, x, y)
        actual, state, info = weak_update(old, x, y, z, 53, 'cpu')
        factors = {c: source[c].L for c in old}
        with patch('chr_training.native.native_factor', side_effect=AssertionError('Recomputed cached factor')):
            cached, cached_state, _ = weak_update(old, x, y, z, 53, 'cpu', factors=factors)
        self.assertEqual(info['steps'], 5000)
        self.assertEqual(info['gamma_2'], .5)
        self.assertEqual(info['mc_rows'], 90)
        self.assertEqual(set(state), set(self.up.NonlinearCompensator(3).state_dict()))
        for c in old:
            torch.testing.assert_close(actual[c].mean, expected[c].mean, rtol=0, atol=0)
            torch.testing.assert_close(actual[c].cov, expected[c].cov, rtol=0, atol=0)
            self.assertEqual(actual[c].count, old[c].count)
            torch.testing.assert_close(cached[c].mean, expected[c].mean, rtol=0, atol=0)
            torch.testing.assert_close(cached[c].cov, expected[c].cov, rtol=0, atol=0)
        for key in state:
            self.assertTrue(torch.equal(state[key], cached_state[key]))


if __name__ == '__main__':
    unittest.main()
