#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import numbers3_live_purchase_audit as purchase_audit
import numbers3_live_reporting as live_reporting
import numbers3_live_scoreboard as live_scoreboard
import numbers3_runner_champion_v4 as v4_runner


def _robust_number3(v) -> str:
    if pd.isna(v):
        return ""
    raw = str(v).strip()
    try:
        n = int(float(raw.replace(",", "")))
        if 0 <= n <= 999:
            return f"{n:03d}"
    except Exception:
        pass
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits[:3] if len(digits) >= 3 else ""


def _install_live_number_parser():
    # CSV inference can turn 037 into integer 37. Keep live scoring correct for leading-zero results.
    purchase_audit.number3 = _robust_number3
    live_scoreboard.number3 = _robust_number3


def main():
    raw_args = list(sys.argv[1:])
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--csv", default="data/numbers3.csv")
    p.add_argument("--pred-dir", default="predictions")
    known, _ = p.parse_known_args(raw_args)

    _install_live_number_parser()
    v4_runner.main()

    pred_dir = Path(known.pred_dir)
    audit_result = purchase_audit.update_purchase_audit(
        Path(known.csv),
        pred_dir / "live_forecast_archive.csv",
        pred_dir / "live_purchase_scoreboard.csv",
        pred_dir / "live_purchase_scoreboard.txt",
    )
    reporting_result = live_reporting.rebuild_live_reports(
        Path(known.csv),
        pred_dir / "live_forecast_archive.csv",
        pred_dir / "live_purchase_scoreboard.csv",
        pred_dir / "live_purchase_scoreboard.txt",
        pred_dir / "cumulative_results.txt",
        list(audit_result.get("missing_issues") or []),
    )
    state = {**audit_result, **reporting_result}
    (pred_dir / "live_purchase_audit_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"live_purchase_audit": state}, ensure_ascii=False))


if __name__ == "__main__":
    main()
