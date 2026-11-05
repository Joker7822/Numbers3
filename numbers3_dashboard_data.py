#!/usr/bin/env python3
"""Build a read-only dashboard snapshot from repository evidence, never guessed values."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
JST = ZoneInfo("Asia/Tokyo")


def json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nullable_float(value):
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def number3(v):
    if pd.isna(v):
        return None
    raw = str(v).strip()
    try:
        n = int(float(raw))
        if 0 <= n <= 999:
            return f"{n:03d}"
    except (ValueError, TypeError, OverflowError):
        pass
    return raw if len(raw) == 3 and raw.isdigit() else None


def main():
    draws_file = ROOT / "data/numbers3.csv"
    if not draws_file.exists():
        raise FileNotFoundError("Restore data/numbers3.csv from data/history_parts first")

    df = pd.read_csv(draws_file)
    df["issue"] = df["回別"].astype(str).str.extract(r"(\d+)").astype(int)
    df["date"] = pd.to_datetime(df["抽せん日"], errors="raise")
    df["number"] = df["本数字"].map(number3)
    if df["number"].isna().any():
        raise ValueError("Unparseable 3-digit winning number")
    df = df.sort_values(["date", "issue"]).drop_duplicates("issue", keep="last")
    last = df.iloc[-1]
    end = last["date"]
    start = end - pd.DateOffset(months=1)
    month = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    month = month.sort_values("date", ascending=False)

    from_second_file = ROOT / "backtests/one_ticket_from_second/historical_tickets.csv"
    history_meta = json_file(ROOT / "backtests/one_ticket_from_second/historical_meta.json")
    replay = {"available": False, "first_issue": None, "last_issue": None}
    historical = pd.DataFrame()
    if from_second_file.exists():
        historical = pd.read_csv(from_second_file, dtype={"actual": str, "top1": str})
        if not historical.empty:
            historical["top1"] = historical["top1"].map(number3)
            historical["actual"] = historical["actual"].map(number3)
            if historical["target_issue"].duplicated().any():
                raise ValueError("Duplicate issue in walk-forward replay")
            if int(historical["target_issue"].iloc[0]) != 2:
                raise ValueError("Historical audit must start at issue 2")
            if not (historical["ticket_count"].eq(1) & historical["ticket_stake"].eq(200)).all():
                raise ValueError("Historical audit contains a non-one-ticket purchase")
            if (historical["target_issue"].diff().iloc[1:] != 1).any():
                raise ValueError("Historical replay has missing draw numbers")
            stake = float(historical["ticket_stake"].sum())
            all_payouts = historical["ticket_return"].notna().all()
            ret = float(historical["ticket_return"].sum()) if all_payouts else None
            replay = {
                "available": True,
                "first_issue": int(historical["target_issue"].iloc[0]),
                "last_issue": int(historical["target_issue"].iloc[-1]),
                "draws": len(historical),
                "hit_count": int(historical["hit1"].sum()),
                "hit_rate": float(historical["hit1"].mean()),
                "random_baseline": 0.001,
                "ticket_count_per_draw": 1,
                "ticket_price_yen": 200,
                "stake": stake,
                "return": ret,
                "profit": (ret - stake) if ret is not None else None,
                "roi": ((ret - stake) / stake) if ret is not None and stake else None,
                "missing_payouts": int(historical["ticket_return"].isna().sum()),
                "exact_nll": nullable_float(historical["exact_nll"].mean()),
                "method": history_meta.get("method"),
                "generated_at": history_meta.get("generated_at"),
            }

    live_file = ROOT / "predictions/live_purchase_scoreboard.csv"
    live_health = json_file(ROOT / "predictions/live_health.json")
    live = pd.read_csv(live_file, dtype={"actual": str, "top20_numbers": str}) if live_file.exists() else pd.DataFrame()
    if not live.empty and "purchase_policy" in live and (live["purchase_policy"] != "top1_straight_one_ticket_per_draw").any():
        raise ValueError("Live purchase audit has not been recalculated under the one-ticket policy")
    live_summary = {"available": False}
    if not live.empty:
        complete = live["total_return"].notna().all()
        stake = float(live["total_stake"].sum())
        ret = float(live["total_return"].sum()) if complete else None
        live_summary = {
            "available": True, "draws": len(live),
            "top1_hits": int(live["top1_hit"].sum()),
            "stake": stake, "return": ret,
            "profit": ret - stake if ret is not None else None,
            "roi": (ret - stake) / stake if ret is not None and stake else None,
            "missing_issues": live_health.get("coverage", {}).get("missing_issues", []),
            "missing_payouts": int(live["total_return"].isna().sum()),
        }

    hist_by_issue = {int(r.target_issue): r for r in historical.itertuples()} if not historical.empty else {}
    live_by_issue = {int(r.target_issue): r for r in live.itertuples()} if not live.empty else {}
    recent = []
    for row in month.itertuples():
        h = hist_by_issue.get(int(row.issue))
        l = live_by_issue.get(int(row.issue))
        live_top1 = (str(l.top20_numbers).split(";")[0].zfill(3) if l is not None else None)
        recent.append({
            "issue": int(row.issue), "date": row.date.date().isoformat(),
            "actual": row.number,
            "walkforward_top1": h.top1 if h is not None else None,
            "walkforward_hit": bool(int(h.hit1)) if h is not None else None,
            "live_top1": live_top1,
            "live_hit": bool(int(l.top1_hit)) if l is not None else None,
        })

    digit_counts = []
    for pos in range(3):
        counts = [0] * 10
        for v in month["number"]:
            counts[int(v[pos])] += 1
        digit_counts.append(counts)

    current = json_file(ROOT / "predictions/numbers3_report_latest.json")
    old_meta = json_file(ROOT / "backtests/historical_meta.json")
    payload = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "source": "Joker7822/Numbers3 main repository snapshot",
        "last_draw": {
            "issue": int(last.issue), "date": last.date.date().isoformat(),
            "number": last.number,
        },
        "period": {
            "start": start.date().isoformat(), "end": end.date().isoformat(),
            "draws": len(month), "definition": "last confirmed draw date minus one calendar month (inclusive)",
        },
        "recent_draws": recent,
        "digit_counts": digit_counts,
        "walkforward_from_second": replay,
        "previous_walkforward": {
            "first_issue": old_meta.get("first_evaluated_issue"),
            "draws": old_meta.get("evaluated_draws"),
        },
        "live": live_summary,
        "forecast": {
            "target_issue": int(current.get("dataset", {}).get("last_issue", "第0回").replace("第", "").replace("回", "")) + 1 if current else None,
            "generated_at": current.get("generated_at"),
            "top_predictions": current.get("top_predictions", [])[:20],
            "edge_status": current.get("validation", {}).get("edge_status"),
            "model_weights": current.get("model_weights", {}),
        },
        "disclaimer": "Past replay is retrospective and not a saved pre-draw live forecast. Buying tickets does not guarantee winnings.",
    }
    output = ROOT / "dashboard/data/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(recent)} draws and {replay.get('draws', 0)} retrospective rows")


if __name__ == "__main__":
    main()
