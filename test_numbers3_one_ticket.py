#!/usr/bin/env python3
"""Regression tests for independent straight/BOX single-ticket simulations."""
import unittest

import numpy as np
import pandas as pd

from numbers3_live_purchase_audit import build_ticket_sets, evaluate_portfolio
from numbers3_historical_backtest import frequency_probs, markov_probs, joint_probs


class OneTicketScenarioTests(unittest.TestCase):
    def test_one_ticket_each_without_combining(self):
        bets = build_ticket_sets(["005", "500", "111", "123"])
        self.assertEqual(bets, {
            "straight": ["005"], "box": ["005"], "set": [], "mini": []
        })
        draw = pd.Series({"ストレート": 78300, "ボックス": 13300})
        scored = evaluate_portfolio(["005", "500", "111", "123"], "500", draw)
        self.assertEqual(scored["straight_tickets"], 1)
        self.assertEqual(scored["box_tickets"], 1)
        self.assertEqual(scored["straight_stake"], 200)
        self.assertEqual(scored["box_stake"], 200)
        self.assertEqual(scored["straight_return"], 0)
        self.assertEqual(scored["box_return"], 13300)
        self.assertEqual(scored["straight_profit"], -200)
        self.assertEqual(scored["box_profit"], 13100)
        # The legacy total is strictly Straight-only, NOT an added 400 yen portfolio.
        self.assertEqual(scored["total_tickets"], 1)
        self.assertEqual(scored["total_stake"], 200)
        self.assertEqual(scored["total_return"], 0)
        self.assertEqual(scored["total_scenario"], "straight_only")

    def test_box_skips_all_identical_digits(self):
        bets = build_ticket_sets(["111", "222", "030", "300"])
        self.assertEqual(bets["straight"], ["111"])
        self.assertEqual(bets["box"], ["003"])
        draw = pd.Series({"ストレート": 95000, "ボックス": 31000})
        scored = evaluate_portfolio(["111", "222", "030", "300"], "300", draw)
        self.assertEqual(scored["straight_ticket_number"], "111")
        self.assertEqual(scored["box_ticket_number"], "030")
        self.assertEqual(scored["straight_hit"], 0)
        self.assertEqual(scored["box_hit"], 1)
        self.assertEqual(scored["box_return"], 31000)

    def test_matching_straight_and_box_wins_are_independent(self):
        draw = pd.Series({"ストレート": 78300, "ボックス": 13300})
        scored = evaluate_portfolio(["005", "500"], "005", draw)
        self.assertEqual(scored["straight_return"], 78300)
        self.assertEqual(scored["box_return"], 13300)
        self.assertEqual(scored["straight_profit"], 78100)
        self.assertEqual(scored["box_profit"], 13100)
        self.assertEqual(scored["total_return"], 78300)

    def test_missing_box_payout_does_not_poison_straight(self):
        draw = pd.Series({"ストレート": 78300, "ボックス": float("nan")})
        scored = evaluate_portfolio(["005"], "005", draw)
        self.assertTrue(np.isnan(scored["box_return"]))
        self.assertTrue(np.isnan(scored["box_roi"]))
        self.assertEqual(scored["straight_return"], 78300)
        self.assertEqual(scored["payout_complete"], 1)

    def test_second_draw_cold_start_uses_first_draw(self):
        first_draw = np.array([[0, 2, 7]], dtype=int)
        f = frequency_probs(first_draw)
        m = markov_probs(first_draw, f)
        p = joint_probs(m)
        self.assertEqual(p.shape, (1000,))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
