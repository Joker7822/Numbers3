#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _money(v) -> str:
    try:
        return f"{int(round(float(v))):,}円"
    except Exception:
        return "n/a"


def _pct(v) -> str:
    try:
        return f"{float(v):+.4%}"
    except Exception:
        return "n/a"


def build_health(pred_dir: Path) -> dict:
    continuity = _read_json(pred_dir / "forecast_continuity.json")
    audit_state = _read_json(pred_dir / "live_purchase_audit_state.json")

    live_path = pred_dir / "live_scoreboard.csv"
    purchase_path = pred_dir / "live_purchase_scoreboard.csv"
    live = pd.read_csv(live_path) if live_path.exists() else pd.DataFrame()
    purchase = pd.read_csv(purchase_path) if purchase_path.exists() else pd.DataFrame()

    missing = [int(x) for x in (audit_state.get("missing_issues") or [])]
    guard_ok = continuity.get("status") == "ok"
    scored = int(len(live))
    purchase_scored = int(len(purchase))
    prospective_opportunities = purchase_scored + len(missing)
    coverage = purchase_scored / prospective_opportunities if prospective_opportunities else None

    if not guard_ok:
        status = "CRITICAL"
        status_reason = "current next-draw forecast continuity check failed"
    elif missing:
        status = "DEGRADED_HISTORY"
        status_reason = "current forecast is safe, but historical live forecast gaps remain"
    else:
        status = "OK"
        status_reason = "current forecast is persisted and no historical live gaps are recorded"

    live_nll = None
    live_rank = None
    if not live.empty:
        live_nll = float(pd.to_numeric(live["exact_nll"], errors="coerce").mean())
        live_rank = float(pd.to_numeric(live["actual_rank"], errors="coerce").mean())

    purchase_metrics = {}
    if not purchase.empty:
        total_stake = float(pd.to_numeric(purchase["total_stake"], errors="coerce").fillna(0).sum())
        returns = pd.to_numeric(purchase["total_return"], errors="coerce")
        total_return = None if returns.isna().any() else float(returns.sum())
        total_profit = None if total_return is None else total_return - total_stake
        total_roi = None if total_profit is None or total_stake <= 0 else total_profit / total_stake
        purchase_metrics = {
            "scored_draws": purchase_scored,
            "any_hit_draws": int(pd.to_numeric(purchase["any_purchase_hit"], errors="coerce").fillna(0).sum()),
            "total_stake": total_stake,
            "total_return": total_return,
            "total_profit": total_profit,
            "total_roi": total_roi,
            "modes": {},
        }
        for mode in ("straight", "box", "set", "mini"):
            stake = float(pd.to_numeric(purchase[f"{mode}_stake"], errors="coerce").fillna(0).sum())
            ret_series = pd.to_numeric(purchase[f"{mode}_return"], errors="coerce")
            ret = None if ret_series.isna().any() else float(ret_series.sum())
            profit = None if ret is None else ret - stake
            roi = None if profit is None or stake <= 0 else profit / stake
            purchase_metrics["modes"][mode] = {
                "hit_draws": int(pd.to_numeric(purchase[f"{mode}_hit"], errors="coerce").fillna(0).sum()),
                "stake": stake,
                "return": ret,
                "profit": profit,
                "roi": roi,
            }

    evidence_status = "INSUFFICIENT_LIVE_SAMPLE"
    if scored >= 500:
        evidence_status = "MATURE_LIVE_SAMPLE"
    elif scored >= 250:
        evidence_status = "INTERMEDIATE_LIVE_SAMPLE"
    elif scored >= 100:
        evidence_status = "INITIAL_LIVE_SAMPLE"

    return {
        "health_version": "live-health-v1",
        "status": status,
        "status_reason": status_reason,
        "continuity": continuity,
        "live": {
            "scored_draws": scored,
            "exact_nll": live_nll,
            "uniform_exact_nll": math.log(1000),
            "nll_improvement_vs_uniform": None if live_nll is None else math.log(1000) - live_nll,
            "mean_actual_rank": live_rank,
            "evidence_status": evidence_status,
        },
        "purchase": purchase_metrics,
        "coverage": {
            "scored_live_draws": purchase_scored,
            "missing_issues": missing,
            "missing_count": len(missing),
            "prospective_opportunities": prospective_opportunities,
            "coverage_rate": coverage,
            "note": "Historical missing issues are not backfilled from replay because that would contaminate prospective evaluation.",
        },
    }


def write_health(pred_dir: Path, out_json: Path, out_txt: Path) -> dict:
    health = build_health(pred_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")

    c = health["continuity"]
    live = health["live"]
    purchase = health.get("purchase") or {}
    coverage = health["coverage"]
    lines = [
        "NUMBERS3 Live Health",
        "=" * 92,
        f"STATUS: {health['status']}",
        f"理由: {health['status_reason']}",
        "",
        "次回予測 continuity",
        "-" * 92,
        f"guard: {c.get('status', 'missing')}",
        f"latest issue: {c.get('latest_issue', 'n/a')} / expected target: {c.get('expected_target_issue', 'n/a')}",
        f"generated: {c.get('latest_generated_at', 'n/a')}",
        f"Champion: {c.get('champion_id', 'n/a')} / generation={c.get('generation', 'n/a')} / config={c.get('config_id', 'n/a')}",
        "",
        "Prospective live evidence",
        "-" * 92,
        f"scored draws: {live['scored_draws']} / evidence={live['evidence_status']}",
        f"Exact NLL: {live['exact_nll']:.8f} / uniform={live['uniform_exact_nll']:.8f} / improvement={live['nll_improvement_vs_uniform']:+.8f}"
        if live["exact_nll"] is not None else "Exact NLL: n/a",
        f"mean actual rank: {live['mean_actual_rank']:.2f}/1000" if live["mean_actual_rank"] is not None else "mean actual rank: n/a",
        "",
        "Virtual purchase audit",
        "-" * 92,
    ]
    if purchase:
        lines += [
            f"scored: {purchase['scored_draws']} / any-hit draws: {purchase['any_hit_draws']}",
            f"stake={_money(purchase['total_stake'])} return={_money(purchase['total_return'])} "
            f"profit={_money(purchase['total_profit'])} ROI={_pct(purchase['total_roi'])}",
        ]
        for mode, m in purchase["modes"].items():
            lines.append(
                f"{mode:8s}: hits={m['hit_draws']} stake={_money(m['stake'])} return={_money(m['return'])} ROI={_pct(m['roi'])}"
            )
    else:
        lines.append("purchase audit: n/a")

    lines += [
        "",
        "Forecast coverage",
        "-" * 92,
        f"coverage: {coverage['scored_live_draws']}/{coverage['prospective_opportunities']} "
        + (f"({coverage['coverage_rate']:.2%})" if coverage['coverage_rate'] is not None else "(n/a)"),
        "historical missing issues: " + (",".join(str(x) for x in coverage["missing_issues"]) if coverage["missing_issues"] else "none"),
        "過去欠番はreplayで埋めません。抽せん前に永続保存された予測だけをprospective成績として扱います。",
        "",
        "判定基準",
        "-" * 92,
        "CRITICAL=current forecast continuity failure / DEGRADED_HISTORY=current safe but historical gaps / OK=no recorded gap.",
        "live n<100は観測段階。NLLやROIが良く見えても予測優位性の証拠とは扱いません。",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return health


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--out-json", default="predictions/live_health.json")
    ap.add_argument("--out-txt", default="predictions/live_health.txt")
    args = ap.parse_args()
    health = write_health(Path(args.pred_dir), Path(args.out_json), Path(args.out_txt))
    print(json.dumps(health, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
