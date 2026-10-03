"""The chaining function of the tail term, hard and soft.

    PYTHONPATH=. ~/bris-env/.venv/bin/python -m unittest tests.test_losses
"""

import math
import unittest

import torch

from xbris.losses import chain_value


class ChainValue(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(-3.0, 12.0, 301, dtype=torch.float64)
        self.t = 2.11

    def test_hard_is_max(self):
        self.assertTrue(torch.equal(chain_value(self.x, self.t, 0.0), self.x.clamp(min=self.t)))

    def test_soft_tends_to_hard(self):
        gap = (chain_value(self.x, self.t, 1e-4) - self.x.clamp(min=self.t)).abs().max()
        self.assertLess(float(gap), 1e-3)

    def test_soft_is_above_hard_and_close_far_from_t(self):
        s = 0.844
        soft, hard = chain_value(self.x, self.t, s), self.x.clamp(min=self.t)
        self.assertTrue(bool((soft >= hard - 1e-12).all()))
        far = (self.x - self.t).abs() > 10 * s
        self.assertLess(float((soft - hard)[far].abs().max()), 1e-3)
        # the most it differs, at the threshold itself, is s * log 2
        self.assertAlmostEqual(float(chain_value(torch.tensor([self.t], dtype=torch.float64), self.t, s)),
                               self.t + s * math.log(2.0), places=9)

    def test_soft_keeps_members_below_t_distinct(self):
        members = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)  # all below t
        hard = chain_value(members, self.t, 0.0)
        soft = chain_value(members, self.t, 0.844)
        self.assertEqual(len(set(hard.tolist())), 1)
        self.assertEqual(len(set(soft.tolist())), 3)

    def test_soft_is_increasing_with_logistic_slope(self):
        x = self.x.clone().requires_grad_(True)
        chain_value(x, self.t, 0.844).sum().backward()
        expected = torch.sigmoid((self.x - self.t) / 0.844)
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-9))

    def test_large_values_do_not_overflow(self):
        big = torch.tensor([1e4, 1e6], dtype=torch.float32)
        out = chain_value(big, self.t, 0.844)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(torch.allclose(out, big))


if __name__ == "__main__":
    unittest.main()
