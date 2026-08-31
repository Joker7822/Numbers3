#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EPS = 1e-15
JST = ZoneInfo("Asia/Tokyo")
SALES_CUTOFF = time(18, 30)


def issue_no(v) -> int:
    m = re.search(r"(\d+)", str(v))
    if not m:
        raise ValueError(f"cannot parse issue: {v!r}")
    return int(m.group(1))


def number3(v) -> str:
    ds = re.findall(r"\d", str(v))
    if len(ds) < 3:
        return ""
    return "".join(ds[:3])


def _eligible_before_draw(generated_at, draw_date) -> bool:
    generated = pd.to_datetime(generated_at, errors="coerce")
    if pd.isna(generated) or pd.isna(draw_date):
        return False
    if generated.tzinfo is None:
        generated = generated.tz_localize(JST)
    else:
        generated = generated.tz_convert(JST)
    cutoff = pd.Timestamp.combine(pd.Timestamp(draw_date).date(), SALES_CUTOFF).tz_localize(JST)
    return generated < cutoff


def update_scoreboard(csv_path: Path, archive_path: Path, out_csv: Path, out_txt: Path) -> dict:
    if not archive_path.exists():
        return {"scored_forecasts": 0, "scored_draws": 0}
    arc = pd.read_csv(archive_path, dtype={"probabilities_json": str})
    if arc.empty:
        return {"scored_forecasts": 0, "scored_draws": 0}
    draws = pd.read_csv(csv_path)
    draws = draws.copy()
    draws["issue"] = draws["回別"].map(issue_no)
    draws["actual"] = draws["本数字"].map(number3)
    draws["draw_date"] = pd.to_datetime(draws["抽せん日"], errors="coerce")
    lookup = {int(r.issue): r for _, r in draws.iterrows()}

    scored = []
    for _, r in arc.iterrows():
        try:
            target = int(r["target_issue"])
        except Exception:
            continue
        d = lookup.get(target)
        if d is None or not d.actual:
            continue
        if not _eligible_before_draw(r.get("generated_at", ""), d.draw_date):
            continue
        try:
            p = np.asarray(json.loads(r["probabilities_json"]), dtype=float)
        except Exception:
            continue
        if p.shape != (1000,):
            continue
        p = np.clip(p, EPS, None)
        p /= p.sum()
        ai = int(d.actual)
        ap = float(p[ai])
        rank = 1 + int(np.sum(p > ap))
        order = np.argsort(p)[::-1]
        scored.append({
            "target_issue": target,
            "draw_date": str(d.draw_date.date()),
            "actual": d.actual,
            "generated_at": r["generated_at"],
            "champion_id": r.get("champion_id", ""),
            "generation": int(r.get("generation", 0)),
            "config_id": r.get("config_id", ""),
            "exact_nll": -math.log(max(ap, EPS)),
            "actual_rank": rank,
            "top1_hit": int(ai == int(order[0])),
            "top20_hit": int(ai in set(order[:20])),
        })

    if not scored:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("NUMBERS3 Live-only leaderboard\nまだ抽せん前に確定したlive予測はありません。\n", encoding="utf-8")
        return {"scored_forecasts": 0, "scored_draws": 0}

    all_scored = pd.DataFrame(scored)
    all_scored["_generated_sort"] = pd.to_datetime(all_scored["generated_at"], errors="coerce", utc=True)
    operational = (
        all_scored.sort_values("_generated_sort")
        .groupby("target_issue", as_index=False).tail(1)
        .drop(columns=["_generated_sort"])
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    operational.sort_values("target_issue").to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [
        "NUMBERS3 Live-only leaderboard", "=" * 88,
        "開発・バックテストに使った過去回ではなく、保存後に実際に抽せんされた予測だけを採点。",
        "同一回に複数Champion予測がある場合、販売締切18:30 JSTより前の最後の予測を採用。", "",
        f"採点済み回数: {len(operational)}",
        f"Exact NLL: {operational.exact_nll.mean():.8f} / 一様={math.log(1000):.8f}",
        f"平均順位: {operational.actual_rank.mean():.2f}/1000",
        f"Top1: {operational.top1_hit.mean():.4%} / Top20: {operational.top20_hit.mean():.4%}", "",
        "世代別", "-" * 88,
    ]
    for gen, g in operational.groupby("generation"):
        lines.append(
            f"Generation {int(gen)}: n={len(g)} ExactNLL={g.exact_nll.mean():.8f} "
            f"Rank={g.actual_rank.mean():.2f} Top1={g.top1_hit.mean():.4%} Top20={g.top20_hit.mean():.4%}"
        )
    lines += ["", "注意: live成績が十分に蓄積するまでは統計的結論を出しません。"]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"scored_forecasts": len(all_scored), "scored_draws": len(operational)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--archive", default="predictions/live_forecast_archive.csv")
    ap.add_argument("--out-csv", default="predictions/live_scoreboard.csv")
    ap.add_argument("--out-txt", default="predictions/live_scoreboard.txt")
    a = ap.parse_args()
    print(json.dumps(update_scoreboard(Path(a.csv), Path(a.archive), Path(a.out_csv), Path(a.out_txt)), ensure_ascii=False))


if __name__ == "__main__":
    main()
