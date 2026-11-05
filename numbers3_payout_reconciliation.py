#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconcile historical ticket payouts from dated, linked public-result records.

Payouts are POST-DRAW evidence only: never use this file in forecast fitting.
The CSV is sourced from a third-party archive, not independently official-verified
for every issue. Keep its provenance visible in dashboard reports.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from numbers3_historical_backtest import write_summary

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data/numbers3_historical_payouts.csv"
DEFAULT_TICKETS = ROOT / "backtests/one_ticket_from_second/historical_tickets.csv"


def load_payout_records(path: Path) -> dict[int, dict]:
    records = pd.read_csv(path, dtype={"actual": str, "date": str, "source_url": str})
    required = {"issue", "date", "actual", "straight_yen", "box_yen", "source_url"}
    if not required.issubset(records.columns):
        raise ValueError(f"Missing payout columns: {sorted(required - set(records.columns))}")
    if records["issue"].duplicated().any():
        raise ValueError("Duplicate issue in payout source")
    result = {}
    for r in records.itertuples(index=False):
        issue = int(r.issue)
        date = str(r.date)
        actual = str(r.actual).zfill(3)
        if issue < 1 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or not re.fullmatch(r"\d{3}", actual):
            raise ValueError(f"Invalid issue/date/actual in payout source: {issue}")
        url = str(r.source_url)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "kujishirube.info":
            raise ValueError(f"Unexpected payout source host for draw {issue}: {url}")
        amounts = []
        for value in (r.straight_yen, r.box_yen):
            amount = float(value)
            if not amount.is_integer() or amount <= 0:
                raise ValueError(f"Invalid payout amount for issue {issue}: {value}")
            amounts.append(int(amount))
        result[issue] = {"issue": issue, "date": date, "actual": actual,
                         "straight_yen": amounts[0], "box_yen": amounts[1],
                         "source_url": url}
    return result


def apply_payout_records(tickets: pd.DataFrame, records: dict[int, dict]) -> tuple[pd.DataFrame, dict]:
    out = tickets.copy()
    if out["target_issue"].duplicated().any():
        raise ValueError("Duplicate historical ticket issue")
    matched = 0
    filled_straight = filled_box = 0
    for i, row in out.iterrows():
        issue = int(row["target_issue"])
        source = records.get(issue)
        if source is None:
            continue
        if str(row["draw_date"]) != source["date"] or str(row["actual"]).zfill(3) != source["actual"]:
            raise ValueError(f"Historical draw mismatch at #{issue}; payout not applied")
        matched += 1
        for mode, column in (("straight", "straight_yen"), ("box", "box_yen")):
            hit = int(row[f"{mode}_ticket_hit"]) == 1
            payout_col = f"{mode}_ticket_return"
            payout = pd.to_numeric(row[payout_col], errors="coerce")
            if not hit and pd.isna(payout):
                raise ValueError(f"Missing zero return for non-winning draw #{issue}")
            if hit and pd.isna(payout):
                payout = source[column]
                out.at[i, payout_col] = payout
                if mode == "straight":
                    filled_straight += 1
                else:
                    filled_box += 1
            if hit and pd.notna(payout) and int(payout) != source[column]:
                raise ValueError(f"Payout source disagrees with existing {mode} prize for #{issue}")
            out.at[i, f"{mode}_ticket_profit"] = (
                float(payout) - float(row[f"{mode}_ticket_stake"]) if pd.notna(payout) else float("nan")
            )
        out.at[i, "ticket_return"] = out.at[i, "straight_ticket_return"]
        out.at[i, "ticket_profit"] = out.at[i, "straight_ticket_profit"]
    missing_straight = int((out["straight_ticket_hit"].eq(1) & out["straight_ticket_return"].isna()).sum())
    missing_box = int((out["box_ticket_hit"].eq(1) & out["box_ticket_return"].isna()).sum())
    return out, {
        "source_type": "third_party_public_result_archive",
        "officially_crosschecked_all_records": False,
        "source_url": "https://kujishirube.info/numbers3/csv/",
        "source_records": len(records), "matched_draws": matched,
        "straight_payouts_recovered": filled_straight,
        "box_payouts_recovered": filled_box,
        "straight_missing_payouts": missing_straight,
        "box_missing_payouts": missing_box,
    }


def reconcile(tickets_path: Path = DEFAULT_TICKETS, source_path: Path = DEFAULT_SOURCE) -> dict:
    records = load_payout_records(source_path)
    tickets = pd.read_csv(tickets_path, dtype={
        "actual": str, "top1": str, "straight_ticket_number": str, "box_ticket_number": str
    })
    if tickets.empty:
        raise ValueError("Empty historical tickets")
    fixed, stats = apply_payout_records(tickets, records)
    fixed.to_csv(tickets_path, index=False, encoding="utf-8-sig")
    meta_path = tickets_path.with_name("historical_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["payout_reconciliation"] = stats
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_summary(tickets_path.with_name("historical_accuracy.txt"), fixed,
                      int(meta.get("warmup_draws", 1)), int(meta.get("refit_interval", 1000)))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.tickets, args.source), ensure_ascii=False, indent=2))
