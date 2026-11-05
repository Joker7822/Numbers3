#!/usr/bin/env python3
"""Regression tests for the one-ticket policy and earliest possible replay."""
import unittest

import numpy as np
import pandas as pd

from numbers3_live_purchase_audit import build_ticket_sets, evaluate_portfolio
from numbers3_historical_backtest import frequency_probs, markov_probs, joint_probs


class OneTicketTests(unittest.TestCase):
    def test_only_one_top_ranked_straight_ticket(self):
        tickets = build_ticket_sets(["005", "500", "111", "123", "456"])
        self.assertEqual(tickets, {
            "straight": ["005"], "box": [], "set": [], "mini": []
        })

    def test_correct_straight_payout_only(self):
        draw = pd.Series({"ストレート": 78300, "ボックス": 13300, "ミニ": 8100})
        won = evaluate_portfolio(["005", "500"], "005", draw)
        self.assertEqual(won["total_tickets"], 1)
        self.assertEqual(won["total_stake"], 200)
        self.assertEqual(won["total_return"], 78300)
        self.assertEqual(won["profit"], 78100)
        self.assertEqual(won["purchase_policy"], "top1_straight_one_ticket_per_draw")
        self.assertEqual(won["box_tickets"], 0)
        lost = evaluate_portfolio(["005", "500"], "500", draw)
        self.assertEqual(lost["total_tickets"], 1)
        self.assertEqual(lost["total_return"], 0)
        self.assertEqual(lost["profit"], -200)

    def test_missing_payout_does_not_fake_zero_return(self):
        draw = pd.Series({"ストレート": float("nan")})
        scored = evaluate_portfolio(["005"], "005", draw)
        self.assertTrue(np.isnan(scored["total_return"]))
        self.assertTrue(np.isnan(scored["roi"]))

    def test_from_second_draw_needs_only_first_draw_history(self):
        first_draw = np.array([[0, 2, 7]], dtype=int)
        f = frequency_probs(first_draw)
        m = markov_probs(first_draw, f)
        p = joint_probs(m)
        self.assertEqual(p.shape, (1000,))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertEqual(len({f"{i:03d}" for i in range(1000)}), 1000)


if __name__ == "__main__":
    unittest.main()
