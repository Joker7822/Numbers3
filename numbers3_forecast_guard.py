#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import numbers3_live_health as live_health

JST = ZoneInfo("Asia/Tokyo")
SALES_CUTOFF = time(18, 30)


def issue_no(v) -> int:
    m = re.search(r"(\d+)", str(v))
    if not m:
        raise ValueError(f"cannot parse issue: {v!r}")
    return int(m.group(1))


def valid_probability_vector(raw) -> bool:
    try:
        p = np.asarray(json.loads(str(raw)), dtype=float)
    except Exception:
        return False
    return bool(
        p.shape == (1000,)
        and np.all(np.isfinite(p))
        and np.all(p >= 0.0)
        and float(p.sum()) > 0.0
    )


def before_candidate_cutoff(generated_at, target_date_candidate) -> bool:
    generated = pd.to_datetime(generated_at, errors="coerce")
    target_date = pd.to_datetime(target_date_candidate, errors="coerce")
    if pd.isna(generated) or pd.isna(target_date):
        return False
    if generated.tzinfo is None:
        generated = generated.tz_localize(JST)
    else:
        generated = generated.tz_convert(JST)
    cutoff = pd.Timestamp.combine(target_date.date(), SALES_CUTOFF).tz_localize(JST)
    return bool(generated < cutoff)


def verify(csv_path: Path, archive_path: Path, out_path: Path) -> dict:
    draws = pd.read_csv(csv_path)
    if draws.empty:
        raise RuntimeError("draw dataset is empty")
    latest_issue = issue_no(draws.iloc[-1]["回別"])
    expected_target = latest_issue + 1

    if not archive_path.exists():
        result = {
            "status": "missing",
            "latest_issue": latest_issue,
            "expected_target_issue": expected_target,
            "valid_forecasts": 0,
            "reason": "live forecast archive does not exist",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    arc = pd.read_csv(archive_path, dtype={"probabilities_json": str})
    target = arc[
        pd.to_numeric(arc.get("target_issue"), errors="coerce").eq(expected_target)
        & pd.to_numeric(arc.get("source_last_issue"), errors="coerce").eq(latest_issue)
    ].copy()

    valid_rows = []
    for _, row in target.iterrows():
        if not valid_probability_vector(row.get("probabilities_json", "")):
            continue
        if not before_candidate_cutoff(row.get("generated_at", ""), row.get("target_date_candidate", "")):
            continue
        valid_rows.append(row)

    if valid_rows:
        valid_rows.sort(key=lambda r: pd.to_datetime(r.get("generated_at"), errors="coerce", utc=True))
        latest = valid_rows[-1]
        result = {
            "status": "ok",
            "latest_issue": latest_issue,
            "expected_target_issue": expected_target,
            "valid_forecasts": len(valid_rows),
            "latest_generated_at": str(latest.get("generated_at", "")),
            "champion_id": str(latest.get("champion_id", "")),
            "generation": int(latest.get("generation", 0)),
            "config_id": str(latest.get("config_id", "")),
            "target_date_candidate": str(latest.get("target_date_candidate", "")),
            "policy": "source issue matches latest draw; 1000 finite probabilities; generated before 18:30 JST candidate-date cutoff",
        }
    else:
        result = {
            "status": "missing",
            "latest_issue": latest_issue,
            "expected_target_issue": expected_target,
            "valid_forecasts": 0,
            "candidate_rows": int(len(target)),
            "reason": "no valid current-target forecast passed continuity checks",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--archive", default="predictions/live_forecast_archive.csv")
    ap.add_argument("--out", default="predictions/forecast_continuity.json")
    args = ap.parse_args()
    out_path = Path(args.out)
    result = verify(Path(args.csv), Path(args.archive), out_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "ok":
        raise SystemExit(2)

    pred_dir = out_path.parent
    health = live_health.write_health(
        pred_dir,
        pred_dir / "live_health.json",
        pred_dir / "live_health.txt",
    )
    print(json.dumps({"live_health": health}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
