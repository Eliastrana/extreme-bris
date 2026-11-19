"""Weighted training draws, against a stand-in with anemoi's __iter__ shape.

    PYTHONPATH=. ~/bris-env/.venv/bin/python -m unittest tests.test_oversample
"""

import unittest

import numpy as np

from xbris.patches import sample_weights, weighted_training_draws


class _Data:
    def __init__(self, n):
        self.dates = np.arange(np.datetime64("2024-01-01T00:00:00"), np.datetime64("2024-01-01T00:00:00") + np.timedelta64(6 * n, "h"), np.timedelta64(6, "h"))


class _Dataset:
    """The parts of EnsNativeGridDataset.__iter__ the patch relies on."""

    def __init__(self, label, n=200, seed=1):
        self.label, self.shuffle = label, True
        self.data = _Data(n)
        self.relative_date_indices = [0, 1]
        self.valid_date_indices = np.arange(n - 1)
        self.chunk_index_range = slice(0, None)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.choice(self.valid_date_indices, size=len(self.valid_date_indices), replace=False)[self.chunk_index_range]
        for i in idx:
            yield int(i)


class Weights(unittest.TestCase):
    def test_target_date_decides(self):
        dates = ["a", "b", "c", "d"]
        p = sample_weights(dates, np.arange(3), [0, 1], {"c": 4.0})
        # index 1 targets date c
        self.assertAlmostEqual(p[1] / p[0], 4.0)

    def test_rollout_takes_the_largest(self):
        dates = ["a", "b", "c", "d"]
        p = sample_weights(dates, np.arange(2), [0, 1, 2], {"d": 3.0, "b": 2.0})
        self.assertAlmostEqual(p[1] / p[0], 3.0 / 2.0)


class Draws(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weighted_training_draws(_Dataset, {"2024-01-01T06:00:00": 50.0})

    def test_training_oversamples(self):
        draws = list(_Dataset("train", seed=3))
        self.assertGreater(draws.count(0), 10)
        self.assertEqual(len(draws), 199)

    def test_validation_is_a_permutation(self):
        draws = list(_Dataset("validation", seed=3))
        self.assertEqual(sorted(draws), list(range(199)))

    def test_same_seed_same_draws(self):
        self.assertEqual(list(_Dataset("train", seed=5)), list(_Dataset("train", seed=5)))

    def test_rng_restored(self):
        d = _Dataset("train")
        real = d.rng
        list(d)
        self.assertIs(d.rng, real)


if __name__ == "__main__":
    unittest.main()
