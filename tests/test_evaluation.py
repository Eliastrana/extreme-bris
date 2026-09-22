from __future__ import annotations

import unittest

import numpy as np

from xbris.evaluation import (
    brier_cases,
    contingency,
    daily_sums,
    deaccumulate,
    fair_brier_cases,
    fair_crps_cases,
    paired_block_bootstrap,
    reliability_table,
    strict_common_mask,
    threshold_weighted_crps_cases,
)


class EvaluationTests(unittest.TestCase):
    def test_fair_crps_two_symmetric_members(self):
        ensemble = np.array([[0.0, 2.0]])
        observation = np.array([1.0])
        self.assertAlmostEqual(float(fair_crps_cases(ensemble, observation)[0]), 0.0)

    def test_threshold_weighted_crps_uses_censoring_transform(self):
        ensemble = np.array([[0.0, 30.0]])
        observation = np.array([25.0])
        score = threshold_weighted_crps_cases(ensemble, observation, 20.0)
        self.assertAlmostEqual(float(score[0]), 0.0)

    def test_fair_brier_removes_member_sampling_term(self):
        ensemble = np.array([[0.0, 30.0, 30.0, 0.0]])
        observation = np.array([30.0])
        self.assertAlmostEqual(float(brier_cases(ensemble, observation, 20.0)[0]), 0.25)
        self.assertAlmostEqual(
            float(fair_brier_cases(ensemble, observation, 20.0)[0]), 1.0 / 6.0
        )

    def test_daily_windows_end_at_30_and_54_hours(self):
        leads = np.arange(0, 61, 6)
        steps = leads.astype(float)[:, None, None]
        sums = daily_sums(steps, leads, [30, 54])
        self.assertEqual(sums.shape, (2, 1, 1))
        self.assertAlmostEqual(float(sums[0, 0, 0]), 12 + 18 + 24 + 30)
        self.assertAlmostEqual(float(sums[1, 0, 0]), 36 + 42 + 48 + 54)

    def test_daily_window_never_uses_analysis_step(self):
        leads = np.arange(0, 31, 6)
        steps = np.ones((len(leads), 1, 1))
        steps[0] = 1000.0
        self.assertAlmostEqual(float(daily_sums(steps, leads, [30])[0, 0, 0]), 4.0)

    def test_deaccumulation(self):
        cumulative = np.array([0.0, 1.0, 3.0, 6.0])[:, None, None]
        step, convention = deaccumulate(cumulative, "cumulative")
        np.testing.assert_allclose(step[:, 0, 0], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(convention, "cumulative")

    def test_reliability_keeps_four_member_probability_levels(self):
        probability = np.array([0.0, 0.25, 0.25, 1.0])
        event = np.array([False, False, True, True])
        rows = reliability_table(probability, event)
        self.assertEqual([row["forecast_probability"] for row in rows], [0.0, 0.25, 1.0])
        self.assertEqual(rows[1]["cases"], 2)
        self.assertAlmostEqual(rows[1]["observed_frequency"], 0.5)

    def test_contingency_counts_false_alarms_over_all_cases(self):
        result = contingency([True, True, False, False], [True, False, True, False])
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["misses"], 1)
        self.assertEqual(result["false_alarms"], 1)
        self.assertEqual(result["correct_negatives"], 1)

    def test_strict_common_mask_requires_every_source_and_member(self):
        observations = np.array([1.0, 2.0, 3.0, np.nan])
        meps = np.array([1.0, 2.0, np.nan, 4.0])
        baseline = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        control = baseline.copy()
        control[1, 1] = np.nan
        tail = baseline.copy()
        common = strict_common_mask(observations, meps, [baseline, control, tail])
        np.testing.assert_array_equal(common, [True, False, False, False])

    def test_block_bootstrap_keeps_constant_paired_difference(self):
        dates = np.arange(
            np.datetime64("2025-08-01"), np.datetime64("2025-08-15"), dtype="datetime64[D]"
        )
        result = paired_block_bootstrap(
            np.full(dates.size, -0.25), dates, block_days=7, replicates=200, seed=7
        )
        self.assertAlmostEqual(result["mean_difference"], -0.25)
        self.assertAlmostEqual(result["ci_lower"], -0.25)
        self.assertAlmostEqual(result["ci_upper"], -0.25)
        self.assertEqual(result["blocks"], 2)


if __name__ == "__main__":
    unittest.main()
