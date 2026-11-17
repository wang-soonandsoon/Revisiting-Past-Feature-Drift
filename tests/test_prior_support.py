"""The original map must supply directions absent from the later observations."""

import unittest
import torch

from chr_core import empty_accumulator, update_accumulator, select_cumulative_chains


class PriorSupportTests(unittest.TestCase):
    def test_unobserved_input_directions_retain_the_native_prior(self):
        dtype = torch.float64
        x = torch.zeros(9, 3, dtype=dtype)
        x[:, 0] = torch.arange(1, 10, dtype=dtype)
        y = torch.zeros_like(x)
        y[:, 1] = x[:, 0]
        ids = torch.arange(3).repeat_interleave(3)
        prior = torch.tensor([[.3, .1, .2], [.7, .5, -.4], [-.2, .6, .9]], dtype=dtype)
        native = [torch.eye(3, dtype=dtype), prior, torch.eye(3, dtype=dtype)]
        features = {0: x, 1: y}
        state = update_accumulator(empty_accumulator(), features, task=2)
        result = select_cumulative_chains(features, ids, native, 2, [1e-4], state)[0]
        self.assertIsNotNone(result['alpha'])
        torch.testing.assert_close(result['map'][1:], prior[1:], atol=1e-12, rtol=0)
        self.assertGreater(float((result['map'][0] - prior[0]).norm()), .1)


if __name__ == '__main__':
    unittest.main()
