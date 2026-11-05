#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for historical one-ticket payout reconciliation."""
import unittest

import pandas as pd

from numbers3_payout_reconciliation import DEFAULT_SOURCE, apply_payout_records, load_payout_records


class HistoricalPayoutTests(unittest.TestCase):
    def setUp(self):
        self.records = load_payout_records(DEFAULT_SOURCE)

    def test_source_covers_45_distinct_draws(self):
        self.assertEqual(len(self.records), 45)
        self.assertEqual(self.records[6395]["actual"], "310")
        self.assertEqual(self.records[6395]["straight_yen"], 58900)
        self.assertEqual(self.records[6395]["box_yen"], 9800)

    def test_recovery_for_straight_and_box_hit(self):
        row = self.row(issue=6395, date="2024-02-01", actual="310", straight_hit=1, box_hit=1)
        fixed, report = apply_payout_records(pd.DataFrame([row]), self.records)
        self.assertEqual(fixed.loc[0, "straight_ticket_return"], 58900)
        self.assertEqual(fixed.loc[0, "box_ticket_return"], 9800)
        self.assertEqual(fixed.loc[0, "ticket_return"], 58900)
        self.assertEqual(report["straight_payouts_recovered"], 1)
        self.assertEqual(report["box_payouts_recovered"], 1)

    def test_box_only_and_nonhit_returns(self):
        row = self.row(issue=136, date="1996-03-12", actual="295", straight_hit=0, box_hit=1)
        fixed, report = apply_payout_records(pd.DataFrame([row]), self.records)
        self.assertEqual(fixed.loc[0, "straight_ticket_return"], 0)
        self.assertEqual(fixed.loc[0, "box_ticket_return"], 12900)
        self.assertEqual(fixed.loc[0, "straight_ticket_profit"], -200)
        self.assertEqual(report["straight_missing_payouts"], 0)

    def test_date_or_number_mismatch_rejected(self):
        row = self.row(issue=6395, date="2024-02-01", actual="311", straight_hit=1, box_hit=1)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            apply_payout_records(pd.DataFrame([row]), self.records)

    def test_missing_winner_preserved_not_guessed(self):
        row = self.row(issue=99999, date="2099-01-01", actual="123", straight_hit=1, box_hit=1)
        fixed, report = apply_payout_records(pd.DataFrame([row]), self.records)
        self.assertTrue(pd.isna(fixed.loc[0, "straight_ticket_return"]))
        self.assertEqual(report["straight_missing_payouts"], 1)
        self.assertEqual(report["box_missing_payouts"], 1)

    @staticmethod
    def row(*, issue, date, actual, straight_hit, box_hit):
        return {
            "target_issue": issue, "draw_date": date, "actual": actual,
            "straight_ticket_hit": straight_hit, "box_ticket_hit": box_hit,
            "straight_ticket_stake": 200, "box_ticket_stake": 200,
            "straight_ticket_return": float("nan") if straight_hit else 0,
            "box_ticket_return": float("nan") if box_hit else 0,
            "straight_ticket_profit": float("nan") if straight_hit else -200,
            "box_ticket_profit": float("nan") if box_hit else -200,
            "ticket_return": float("nan") if straight_hit else 0,
            "ticket_profit": float("nan") if straight_hit else -200,
        }


if __name__ == "__main__":
    unittest.main()
