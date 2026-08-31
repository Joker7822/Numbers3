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
    straight = unique_preserve(top20)
    box = unique_preserve(["".join(sorted(n)) for n in top20 if len(set(n)) > 1])
    set_bets = unique_preserve([n for n in top20 if len(set(n)) > 1])
    mini = unique_preserve([n[-2:] for n in top20])
    return {"straight": straight, "box": box, "set": set_bets, "mini": mini}


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
    result: dict[str, object] = {}
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
                    payout_complete = False
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
        total_tickets += len(bets)
        if mode_complete:
            total_return += mode_return
        if hits:
            hit_modes.append(mode)

    total_stake = total_tickets * UNIT_PRICE
    profit = total_return - total_stake if payout_complete else np.nan
    roi = profit / total_stake if payout_complete and total_stake else np.nan
    result.update({
        "purchase_policy": "top20_unique_bets_all_modes_once",
        "unit_price": UNIT_PRICE,
        "total_tickets": total_tickets,
        "total_stake": total_stake,
        "total_return": total_return if payout_complete else np.nan,
        "profit": profit,
        "roi": roi,
        "any_purchase_hit": int(bool(hit_modes)),
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
        "仮想購入: Top20から重複する同一賭式を除き、ストレート/ボックス/セット/ミニを各1口。",
        f"1口={UNIT_PRICE}円。ボックス/セットは3桁同一数字を購入対象外。ROI=(払戻-購入額)/購入額。", "",
    ]

    if operational.empty:
        lines.append("照合済み通常予測はありません。")
    else:
        n = len(operational)
        total_stake = float(operational["total_stake"].sum())
        complete = operational["payout_complete"].astype(int).eq(1).all()
        total_return = float(operational["total_return"].sum()) if complete else np.nan
        total_profit = total_return - total_stake if complete else np.nan
        total_roi = total_profit / total_stake if complete and total_stake else np.nan
        lines += [
            "集計", "-" * 112,
            f"採点済み回数: {n}",
            f"Exact NLL: {operational['exact_nll'].mean():.8f} / 一様={math.log(1000):.8f}",
            f"平均実数字順位: {operational['actual_rank'].mean():.2f}/1000",
            f"Straight Top1: {operational['top1_hit'].mean():.4%} / Top20: {operational['top20_straight_hit'].mean():.4%}",
            f"購入方式で何らか的中: {int(operational['any_purchase_hit'].sum())}/{n} ({operational['any_purchase_hit'].mean():.4%})",
            f"Straight的中回率: {operational['straight_hit'].mean():.4%}",
            f"BOX的中回率: {operational['box_hit'].mean():.4%}",
            f"SET的中回率: {operational['set_hit'].mean():.4%}",
            f"MINI的中回率: {operational['mini_hit'].mean():.4%}",
            f"仮想購入額合計: {int(round(total_stake)):,}円",
            f"払戻合計: {int(round(total_return)):,}円" if complete else "払戻合計: 一部払戻データ欠損のため算出不可",
            f"収支: {int(round(total_profit)):+,}円" if complete else "収支: 算出不可",
            f"ROI: {total_roi:+.4%}" if complete else "ROI: 算出不可",
            f"全口外れ非掲載回数: {int((operational['any_purchase_hit'] == 0).sum())}", "",
        ]

        lines += ["的中回のみ（全口外れは非掲載）", "-" * 112]
        hit_rows = operational[operational["any_purchase_hit"].astype(int).eq(1)]
        if hit_rows.empty:
            lines.append("該当なし")
        else:
            for _, r in hit_rows.iterrows():
                roi = r["roi"]
                roi_text = f"{float(roi):+.2%}" if pd.notna(roi) else "算出不可"
                ret_text = f"{int(round(float(r['total_return']))):,}円" if pd.notna(r["total_return"]) else "払戻不明"
                profit_text = f"{int(round(float(r['profit']))):+,}円" if pd.notna(r["profit"]) else "算出不可"
                lines.append(
                    f"第{int(r['target_issue'])}回 {r['draw_date']} actual={r['actual']} rank={int(r['actual_rank'])} "
                    f"hits={r['hit_modes'] or '-'} tickets={int(r['total_tickets'])} stake={int(r['total_stake']):,}円 "
                    f"return={ret_text} profit={profit_text} ROI={roi_text} NLL={float(r['exact_nll']):.8f}"
                )
                lines.append(f"  {r['hit_details']}")

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
