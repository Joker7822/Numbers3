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

from numbers3_hit_rate_lab import best_box_by_mass, box_hit, shadow_comparison

JST = ZoneInfo("Asia/Tokyo")
EPS = 1e-15
UNIT_PRICE = 200
SALES_CUTOFF = time(18, 30)
PAYOUT_COLUMNS = {
    "straight": "ストレート",
    "box": "ボックス",
    "set_straight": "セット(ストレート)",
    "set_box": "セット(ボックス)",
    "mini": "ミニ",
}


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


def payout_value(row: pd.Series, column: str) -> float | None:
    value = row.get(column, np.nan)
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def cutoff_timestamp(draw_date) -> pd.Timestamp:
    d = pd.Timestamp(draw_date).date()
    return pd.Timestamp.combine(d, SALES_CUTOFF).tz_localize(JST)


def generated_timestamp(v) -> pd.Timestamp | None:
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(JST)
    return ts.tz_convert(JST)


def is_eligible_forecast(generated_at, draw_date) -> bool:
    ts = generated_timestamp(generated_at)
    if ts is None or pd.isna(draw_date):
        return False
    return ts < cutoff_timestamp(draw_date)


def top20_numbers(p: np.ndarray) -> list[str]:
    order = np.argsort(p)[::-1][:20]
    return [f"{int(i):03d}" for i in order]


def unique_preserve(values):
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def build_ticket_sets(top20: list[str]) -> dict[str, list[str]]:
    """Two independent simulations: one straight ticket and one valid box ticket.

    No combined purchase is assumed. BOX rejects three identical digits; if the
    top-ranked number is such a triple, choose the next eligible ranked number.
    """
    if not top20:
        raise ValueError("top20 must contain at least one prediction")
    box_candidate = next((n for n in top20 if len(set(n)) > 1), None)
    if box_candidate is None:
        raise ValueError("no eligible box candidate among ranked forecasts")
    return {
        "straight": [top20[0]],
        "box": ["".join(sorted(box_candidate))],
        "set": [],
        "mini": [],
    }


def ticket_return(mode: str, bet: str, actual: str, draw: pd.Series) -> tuple[float | None, bool]:
    if mode == "straight":
        hit = bet == actual
        col = PAYOUT_COLUMNS["straight"]
    elif mode == "box":
        hit = "".join(sorted(actual)) == bet
        col = PAYOUT_COLUMNS["box"]
    elif mode == "set":
        if bet == actual:
            hit = True
            col = PAYOUT_COLUMNS["set_straight"]
        elif sorted(bet) == sorted(actual):
            hit = True
            col = PAYOUT_COLUMNS["set_box"]
        else:
            hit = False
            col = PAYOUT_COLUMNS["set_box"]
    elif mode == "mini":
        hit = bet == actual[-2:]
        col = PAYOUT_COLUMNS["mini"]
    else:
        raise ValueError(mode)

    if not hit:
        return 0.0, False
    payout = payout_value(draw, col)
    return payout, True


def evaluate_portfolio(top20: list[str], actual: str, draw: pd.Series) -> dict:
    tickets = build_ticket_sets(top20)
    result: dict[str, object] = {
        "straight_ticket_number": top20[0],
        "box_ticket_number": next(n for n in top20 if len(set(n)) > 1),
    }
    # This canonical total is the STRAIGHT scenario only (legacy-safe alias).
    # BOX is reported independently and must never be added to this total.
    total_tickets = 0
    total_return = 0.0
    payout_complete = True
    hit_modes: list[str] = []
    hit_details: list[str] = []

    for mode in ("straight", "box", "set", "mini"):
        bets = tickets[mode]
        mode_return = 0.0
        mode_complete = True
        hits = []
        for bet in bets:
            ret, hit = ticket_return(mode, bet, actual, draw)
            if hit:
                hits.append(bet)
                if ret is None:
                    mode_complete = False
                    hit_details.append(f"{mode}:{bet}=払戻不明")
                else:
                    mode_return += ret
                    hit_details.append(f"{mode}:{bet}={int(round(ret))}円")
        stake = len(bets) * UNIT_PRICE
        result[f"{mode}_tickets"] = len(bets)
        result[f"{mode}_stake"] = stake
        result[f"{mode}_return"] = mode_return if mode_complete else np.nan
        result[f"{mode}_hit"] = int(bool(hits))
        result[f"{mode}_winning_bets"] = ";".join(hits)
        result[f"{mode}_profit"] = mode_return - stake if mode_complete else np.nan
        result[f"{mode}_roi"] = ((mode_return - stake) / stake
                                 if mode_complete and stake else np.nan)
        if mode == "straight":
            total_tickets = len(bets)
            if mode_complete:
                total_return = mode_return
            else:
                payout_complete = False
        if hits:
            hit_modes.append(mode)

    total_stake = total_tickets * UNIT_PRICE
    profit = total_return - total_stake if payout_complete else np.nan
    roi = profit / total_stake if payout_complete and total_stake else np.nan
    result.update({
        "purchase_policy": "independent_one_ticket_straight_vs_box",
        "unit_price": UNIT_PRICE,
        "total_scenario": "straight_only",
        "total_tickets": total_tickets,
        "total_stake": total_stake,
        "total_return": total_return if payout_complete else np.nan,
        "profit": profit,
        "roi": roi,
        "any_purchase_hit": int(bool(hit_modes)),  # either scenario, NOT simultaneous purchase
        "hit_modes": ";".join(hit_modes),
        "hit_details": " / ".join(hit_details),
        "payout_complete": int(payout_complete),
    })
    return result


def update_purchase_audit(csv_path: Path, archive_path: Path, out_csv: Path, out_txt: Path) -> dict:
    draws = pd.read_csv(csv_path).copy()
    draws["issue"] = draws["回別"].map(issue_no)
    draws["actual"] = draws["本数字"].map(number3)
    draws["draw_date"] = pd.to_datetime(draws["抽せん日"], errors="coerce")
    draws = draws.sort_values(["issue", "draw_date"]).drop_duplicates("issue", keep="last")

    if not archive_path.exists():
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("NUMBERS3 通常予測 購入方式別Live照合\n予測アーカイブがありません。\n", encoding="utf-8")
        return {"scored_draws": 0, "missing_issues": []}

    arc = pd.read_csv(archive_path, dtype={"probabilities_json": str})
    if arc.empty:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("NUMBERS3 通常予測 購入方式別Live照合\n予測アーカイブが空です。\n", encoding="utf-8")
        return {"scored_draws": 0, "missing_issues": []}

    candidates = []
    draw_lookup = {int(r.issue): r for _, r in draws.iterrows()}
    for _, r in arc.iterrows():
        try:
            target = int(r["target_issue"])
        except Exception:
            continue
        d = draw_lookup.get(target)
        if d is None or not d.actual or pd.isna(d.draw_date):
            continue
        if not is_eligible_forecast(r.get("generated_at", ""), d.draw_date):
            continue
        try:
            p = np.asarray(json.loads(r["probabilities_json"]), dtype=float)
        except Exception:
            continue
        if p.shape != (1000,) or not np.all(np.isfinite(p)):
            continue
        p = np.clip(p, EPS, None)
        p /= p.sum()
        ai = int(d.actual)
        ap = float(p[ai])
        rank = 1 + int(np.sum(p > ap))
        top20 = top20_numbers(p)
        purchase = evaluate_portfolio(top20, d.actual, d)
        # Retrospective POLICY comparisons on an archived, pre-draw joint
        # distribution. These were not precommitted Live purchase decisions.
        mass_choice, mass_probability = best_box_by_mass(p, distinct_only=False)
        unique_choice, unique_probability = best_box_by_mass(p, distinct_only=True)
        gen = pd.to_numeric(r.get("generation", 0), errors="coerce")
        candidates.append({
            "target_issue": target,
            "draw_date": str(d.draw_date.date()),
            "actual": d.actual,
            "generated_at": str(r.get("generated_at", "")),
            "champion_id": str(r.get("champion_id", "")),
            "generation": int(gen) if pd.notna(gen) else 0,
            "config_id": str(r.get("config_id", "")),
            "actual_probability": ap,
            "exact_nll": -math.log(max(ap, EPS)),
            "actual_rank": rank,
            "top1_hit": int(ai == int(np.argmax(p))),
            "top20_straight_hit": int(d.actual in set(top20)),
            "top20_numbers": ";".join(top20),
            "box_mass_number": mass_choice,
            "box_mass_hit": box_hit(mass_choice, d.actual),
            "box_mass_model_probability": mass_probability,
            "box_unique_number": unique_choice,
            "box_unique_hit": box_hit(unique_choice, d.actual),
            "box_unique_model_probability": unique_probability,
            **purchase,
        })

    if candidates:
        all_rows = pd.DataFrame(candidates)
        all_rows["_generated_sort"] = pd.to_datetime(all_rows["generated_at"], errors="coerce", utc=True)
        operational = (
            all_rows.sort_values("_generated_sort")
            .groupby("target_issue", as_index=False)
            .tail(1)
            .drop(columns=["_generated_sort"])
            .sort_values("target_issue")
        )
    else:
        operational = pd.DataFrame()

    arc_targets = pd.to_numeric(arc.get("target_issue", pd.Series(dtype=float)), errors="coerce").dropna().astype(int)
    missing: list[int] = []
    if not arc_targets.empty:
        start_issue = int(arc_targets.min())
        actual_issues = [int(x) for x in draws.loc[draws["issue"] >= start_issue, "issue"].tolist()]
        scored_set = set(operational["target_issue"].astype(int)) if not operational.empty else set()
        missing = [x for x in actual_issues if x not in scored_set]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    operational.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines = [
        "NUMBERS3 通常予測 購入方式別Live照合", "=" * 112,
        "対象: 抽せん前に保存された通常予測。過去へ再適用するChampion replayとは別集計。",
        "採用予測: 各回について販売締切(18:30 JST)より前に保存された最後の予測。",
        "仮想購入: ストレート1口とボックス1口を別々に検証し、同時購入にはしない。",
        f"各方式1回につき1口={UNIT_PRICE}円。BOXはゾロ目不可のため次の購入可能な上位候補を採用。", "",
    ]

    if operational.empty:
        lines.append("照合済み通常予測はありません。")
    else:
        n = len(operational)
        lines += [
            "方式別成績（互いに独立した1口購入シナリオ）", "-" * 112,
            f"照合済み回数: {n}",
            f"Exact NLL: {operational['exact_nll'].mean():.8f} / 一様={math.log(1000):.8f}",
            f"平均実数字順位: {operational['actual_rank'].mean():.2f}/1000",
        ]
        for mode, label in (("straight", "ストレート"), ("box", "ボックス")):
            stake = float(operational[f"{mode}_stake"].sum())
            complete = operational[f"{mode}_return"].notna().all()
            ret = float(operational[f"{mode}_return"].sum()) if complete else np.nan
            profit = ret - stake if complete else np.nan
            roi = profit / stake if complete and stake else np.nan
            lines += [
                f"{label}のみ1口: 的中 {int(operational[f'{mode}_hit'].sum())}/{n}"
                f" ({operational[f'{mode}_hit'].mean():.4%})",
                f"  購入額={int(stake):,}円 / 払戻="
                + (f"{int(ret):,}円 / 損益={int(profit):+,}円 / ROI={roi:+.4%}"
                   if complete else "欠損あり / 損益・ROI=算出不可"),
            ]
        lines += ["", "方式別の的中回（全回はCSVを参照）", "-" * 112]
        hit_rows = operational[operational["any_purchase_hit"].astype(int).eq(1)]
        if hit_rows.empty:
            lines.append("該当なし")
        else:
            for _, row in hit_rows.iterrows():
                lines.append(
                    f"第{int(row['target_issue'])}回 {row['draw_date']} 実={row['actual']} "
                    f"Straight={row['straight_ticket_number']} "
                    f"hit={int(row['straight_hit'])} 払戻={row['straight_return']}円 / "
                    f"BOX={row['box_ticket_number']} hit={int(row['box_hit'])} "
                    f"払戻={row['box_return']}円"
                )

    if not operational.empty:
        comp = shadow_comparison(operational)
        lines += [
            "", "BOX候補政策のシャドー評価（抽せん前保存確率への事後設計政策の適用）", "-" * 112,
            f"既存方式: {comp['modes']['baseline']['hits']}/{n}",
            f"並べ替え質量最大: {comp['modes']['mass']['hits']}/{n} / paired p={comp['modes']['mass']['paired_p_exploratory']:.4f}",
            f"3桁すべて異なる: {comp['modes']['unique']['hits']}/{n} / paired p={comp['modes']['unique']['paired_p_exploratory']:.4f}",
            "これは過去のLive予測数字そのものではなく、保存済み確率への再解析。新政策の将来優位性は未確認。",
        ]
    lines += ["", "予測欠番監査", "-" * 112]
    if missing:
        lines.append("WARNING: 抽せん結果は存在するが、販売締切前の通常予測アーカイブが無い回があります。")
        lines.append("missing_issues=" + ",".join(str(x) for x in missing))
    else:
        lines.append("missing_issues=none")
    lines += [
        "",
        "注意: 仮想購入は固定監査ポリシーであり、実際に購入したことを意味しません。",
        "当せん金は各回のCSV記録値を使用し、払戻データが欠ける的中回はROIを算出しません。",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"scored_draws": int(len(operational)), "missing_issues": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--archive", default="predictions/live_forecast_archive.csv")
    ap.add_argument("--out-csv", default="predictions/live_purchase_scoreboard.csv")
    ap.add_argument("--out-txt", default="predictions/live_purchase_scoreboard.txt")
    a = ap.parse_args()
    print(json.dumps(update_purchase_audit(Path(a.csv), Path(a.archive), Path(a.out_csv), Path(a.out_txt)), ensure_ascii=False))


if __name__ == "__main__":
    main()
