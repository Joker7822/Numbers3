#!/usr/bin/env python3
"""Safety and mathematical regression tests for shadow BOX policies."""
import unittest
import numpy as np
import pandas as pd

from numbers3_hit_rate_lab import (
    BOX_KEYS, BOX_DISTINCT_MASK, best_box_by_mass, box_hit,
    paired_exact_p, shadow_comparison,
)


class BoxHitRateLabTests(unittest.TestCase):
    def test_complete_ticket_space(self):
        self.assertEqual(len(BOX_KEYS), 210)
        self.assertEqual(int(BOX_DISTINCT_MASK.sum()), 120)
        self.assertNotIn("111", BOX_KEYS)
        self.assertIn("112", BOX_KEYS)
        self.assertIn("123", BOX_KEYS)

    def test_uniform_distribution_favors_six_permutation_group(self):
        p = np.ones(1000)
        for distinct in (True, False):
            pick, mass = best_box_by_mass(p, distinct_only=distinct)
            self.assertEqual(len(set(pick)), 3)
            self.assertAlmostEqual(mass, .006)

    def test_sums_permutations_instead_of_only_ranking_one_number(self):
        p = np.zeros(1000)
        # The individual winner is 125, but 1-2-3 has the greater total mass.
        p[125] = .30
        for n in (123,132,213,231,312,321):
            p[n] = .10
        for distinct in (True, False):
            pick, mass = best_box_by_mass(p, distinct_only=distinct)
            self.assertEqual(pick, "123")
            self.assertAlmostEqual(mass, 2/3)

    def test_double_can_win_unrestricted_mass_but_not_distinct(self):
        p = np.zeros(1000)
        for n in (112,121,211):
            p[n] = .25
        p[123] = .25
        best, _ = best_box_by_mass(p, distinct_only=False)
        safe, _ = best_box_by_mass(p, distinct_only=True)
        self.assertEqual(best, "112")
        self.assertEqual(safe, "123")

    def test_invalid_distribution_rejected(self):
        for p in (np.ones(999), np.zeros(1000),
                  np.full(1000, float("nan")), np.full(1000, -1.0)):
            with self.assertRaises(ValueError):
                best_box_by_mass(p, distinct_only=True)

    def test_permutations_are_equivalent(self):
        self.assertEqual(box_hit("012", "201"), 1)
        self.assertEqual(box_hit("001", "100"), 1)
        self.assertEqual(box_hit("001", "011"), 0)

    def test_paired_exact_statistic_does_not_fake_significance(self):
        self.assertEqual(paired_exact_p(0,0), 1)
        self.assertGreater(paired_exact_p(5, 8), .05)
        self.assertAlmostEqual(paired_exact_p(0, 5), 0.0625)

    def test_comparison_keeps_original_and_shadows_separate(self):
        df = pd.DataFrame([
            {"target_issue":5000,"box_ticket_hit":1,"box_mass_hit":0,"box_unique_hit":1},
            {"target_issue":5001,"box_ticket_hit":0,"box_mass_hit":1,"box_unique_hit":1},
            {"target_issue":5002,"box_ticket_hit":0,"box_mass_hit":0,"box_unique_hit":0},
        ])
        report = shadow_comparison(df)
        self.assertEqual(report["modes"]["baseline"]["hits"], 1)
        self.assertEqual(report["modes"]["mass"]["hits"], 1)
        self.assertEqual(report["modes"]["unique"]["hits"], 2)
        self.assertTrue(report["retrospective_policy_design"])
        self.assertTrue(report["production_policy_unchanged"])


if __name__ == "__main__":
    unittest.main()
