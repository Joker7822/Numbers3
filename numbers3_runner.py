#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NUMBERS3 daily autonomous runner.

1) Optionally refresh official draw data through scrapingnumbers3.py.
2) Resolve all unresolved historical predictions by draw issue number.
3) Run the evolutionary ensemble/backtest.
4) Append one cumulative prediction set for the next issue.
5) Append backtest metrics by source issue.
6) Render human-readable latest/cumulative .txt reports.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from numbers3_evo_agent import (
    AgentConfig,
    parse_number,
    preserve_merge_scraped,
    run_agent,
)

HISTORY_COLUMNS = [
    "generation",
    "generated_at",
    "source_last_issue",
    "source_last_draw_date",
    "target_issue",
    "target_date_candidate",
    "rank",
    "prediction",
    "model_probability",
    "uniform_probability",
    "relative_to_uniform",
    "status",
    "actual_draw_date",
    "winning_number",
    "match_types",
    "prize_details",
    "max_applicable_prize",
]

BACKTEST_COLUMNS = [
    "generation",
    "generated_at",
    "source_last_issue",
    "source_last_draw_date",
    "validation_draws",
    "validation_period_start",
    "validation_period_end",
    "ensemble_logloss",
    "uniform_logloss",
    "improvement_vs_uniform",
    "edge_status",
    "top1_straight_hit_rate",
    "top20_straight_hit_rate",
    "weight_frequency",
    "weight_markov",
    "weight_logreg",
    "weight_extra_trees",
]


def issue_no(v) -> int:
    m = re.search(r"(\d+)", str(v))
    if not m:
        raise ValueError(f"回号を解釈できません: {v!r}")
    return int(m.group(1))


def yen(v) -> str:
    if pd.isna(v):
        return "確認できません"
    try:
        return f"{int(round(float(v))):,}円"
    except Exception:
        return "確認できません"


def load_draws(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["_issue_no"] = df["回別"].map(issue_no)
    df["_number"] = ["".join(map(str, parse_number(v))) for v in df["本数字"]]
    df["_date"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.sort_values(["_issue_no", "_date"]).drop_duplicates("_issue_no", keep="last")
    return df


def read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path, dtype={"prediction": str, "winning_number": str})
    for c in HISTORY_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    if "prediction" in df:
        df["prediction"] = df["prediction"].fillna("").astype(str).str.zfill(3)
    if "winning_number" in df:
        df["winning_number"] = df["winning_number"].fillna("").astype(str)
        df.loc[df["winning_number"].ne(""), "winning_number"] = (
            df.loc[df["winning_number"].ne(""), "winning_number"].str.zfill(3)
        )
    return df[HISTORY_COLUMNS]


def applicable_awards(pred: str, actual: str, row: pd.Series) -> tuple[List[str], List[str], float | None]:
    exact = pred == actual
    box = sorted(pred) == sorted(actual)
    mini = pred[-2:] == actual[-2:]

    matches: List[str] = []
    prizes: List[str] = []
    amounts: List[float] = []

    def add(label: str, col: str):
        matches.append(label)
        value = row.get(col, np.nan)
        prizes.append(f"{label}={yen(value)}")
        if pd.notna(value):
            try:
                amounts.append(float(value))
            except Exception:
                pass

    if exact:
        add("ストレート", "ストレート")
        if pd.notna(row.get("ボックス", np.nan)):
            add("ボックス", "ボックス")
        if pd.notna(row.get("セット(ストレート)", np.nan)):
            add("セット(ストレート)", "セット(ストレート)")
    elif box:
        add("ボックス", "ボックス")
        if pd.notna(row.get("セット(ボックス)", np.nan)):
            add("セット(ボックス)", "セット(ボックス)")

    if mini:
        add("ミニ", "ミニ")

    if not matches:
        return ["はずれ"], ["当選なし"], None
    return matches, prizes, max(amounts) if amounts else None


def resolve_history(history: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    lookup = {int(r["_issue_no"]): r for _, r in draws.iterrows()}

    for idx, h in history.iterrows():
        if str(h.get("status", "")) == "照合済":
            continue
        try:
            target = int(h["target_issue"])
        except Exception:
            continue
        if target not in lookup:
            history.at[idx, "status"] = "未抽せん"
            continue

        row = lookup[target]
        pred = str(h["prediction"]).zfill(3)
        actual = str(row["_number"]).zfill(3)
        matches, prizes, max_prize = applicable_awards(pred, actual, row)

        history.at[idx, "status"] = "照合済"
        history.at[idx, "actual_draw_date"] = str(pd.to_datetime(row["抽せん日"]).date())
        history.at[idx, "winning_number"] = actual
        history.at[idx, "match_types"] = " / ".join(matches)
        history.at[idx, "prize_details"] = " / ".join(prizes)
        history.at[idx, "max_applicable_prize"] = "" if max_prize is None else int(round(max_prize))
    return history


def append_predictions(history: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, bool]:
    source_issue = issue_no(report["dataset"]["last_issue"])
    target_issue = source_issue + 1

    existing = history[
        (pd.to_numeric(history["source_last_issue"], errors="coerce") == source_issue)
        & (pd.to_numeric(history["target_issue"], errors="coerce") == target_issue)
    ]
    if not existing.empty:
        return history, False

    rows = []
    for p in report["top_predictions"]:
        rows.append(
            {
                "generation": report["generation"],
                "generated_at": report["generated_at"],
                "source_last_issue": source_issue,
                "source_last_draw_date": report["dataset"]["last_draw_date"],
                "target_issue": target_issue,
                "target_date_candidate": report["dataset"]["model_next_weekday_candidate"],
                "rank": p["rank"],
                "prediction": str(p["number"]).zfill(3),
                "model_probability": p["probability"],
                "uniform_probability": 0.001,
                "relative_to_uniform": p["probability"] / 0.001,
                "status": "未抽せん",
                "actual_draw_date": "",
                "winning_number": "",
                "match_types": "",
                "prize_details": "",
                "max_applicable_prize": "",
            }
        )
    return pd.concat([history, pd.DataFrame(rows)], ignore_index=True), True


def append_backtest(path: Path, report: dict) -> bool:
    source_issue = issue_no(report["dataset"]["last_issue"])
    if path.exists():
        bt = pd.read_csv(path)
    else:
        bt = pd.DataFrame(columns=BACKTEST_COLUMNS)

    if not bt.empty and source_issue in set(pd.to_numeric(bt["source_last_issue"], errors="coerce").dropna().astype(int)):
        return False

    v = report["validation"]
    w = report["model_weights"]
    row = {
        "generation": report["generation"],
        "generated_at": report["generated_at"],
        "source_last_issue": source_issue,
        "source_last_draw_date": report["dataset"]["last_draw_date"],
        "validation_draws": v["draws"],
        "validation_period_start": v["period_start"],
        "validation_period_end": v["period_end"],
        "ensemble_logloss": v["ensemble_logloss"],
        "uniform_logloss": v["uniform_logloss"],
        "improvement_vs_uniform": v["logloss_improvement_vs_uniform"],
        "edge_status": v["edge_status"],
        "top1_straight_hit_rate": v["top1_straight_hit_rate"],
        "top20_straight_hit_rate": v["top20_straight_hit_rate"],
        "weight_frequency": w.get("frequency", 0.0),
        "weight_markov": w.get("markov", 0.0),
        "weight_logreg": w.get("logreg", 0.0),
        "weight_extra_trees": w.get("extra_trees", 0.0),
    }
    bt = pd.concat([bt, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def render_latest(path: Path, report: dict):
    d = report["dataset"]
    v = report["validation"]
    w = report["model_weights"]
    target_issue = issue_no(d["last_issue"]) + 1

    lines = [
        "NUMBERS3 最新予測",
        "=" * 64,
        f"生成日時: {report['generated_at']}",
        f"学習済み最終抽せん: {d['last_issue']} / {d['last_draw_date']} / 本数字 {d['last_number']}",
        f"予測対象: 第{target_issue}回",
        f"抽せん日候補: {d['model_next_weekday_candidate']}（回号を主キーに後日照合）",
        f"世代: {report['generation']}",
        "",
        "バックテスト",
        "-" * 64,
        f"検証期間: {v['period_start']} ～ {v['period_end']} / {v['draws']}回",
        f"Ensemble log loss: {v['ensemble_logloss']:.8f}",
        f"Uniform log loss : {v['uniform_logloss']:.8f}",
        f"差(一様 - model): {v['logloss_improvement_vs_uniform']:+.8f}",
        f"優位性判定: {v['edge_status']}",
        f"Top1 ストレート包含率: {v['top1_straight_hit_rate']:.4%}",
        f"Top20 ストレート包含率: {v['top20_straight_hit_rate']:.4%}",
        "",
        "モデル重み",
        "-" * 64,
    ]
    for name, weight in sorted(w.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"{name:12s}: {weight:.6%}")

    lines += ["", "予測ランキング", "-" * 64]
    for p in report["top_predictions"]:
        lines.append(
            f"{int(p['rank']):02d}. {str(p['number']).zfill(3)}  "
            f"model={float(p['probability']):.6%}  "
            f"vs_uniform={float(p['probability'])/0.001:.3f}x"
        )

    lines += [
        "",
        "注意",
        "-" * 64,
        "・当選を保証するものではありません。",
        "・当選金額は購入方式（ストレート/ボックス/セット/ミニ）で異なります。",
        "・抽せん後は predictions/cumulative_results.txt に方式別の照合結果を追記します。",
        "・一様分布よりバックテストが良くない場合、予測優位性は確認できません。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render_cumulative(path: Path, history: pd.DataFrame):
    lines = [
        "NUMBERS3 予測結果・累積照合",
        "=" * 72,
        "各候補は購入方式別に照合します。表示金額はその方式で購入した場合の払戻金です。",
        "",
    ]
    if history.empty:
        lines.append("予測履歴はまだありません。")
    else:
        work = history.copy()
        work["target_issue_num"] = pd.to_numeric(work["target_issue"], errors="coerce")
        for target, grp in work.sort_values(["target_issue_num", "rank"], ascending=[False, True]).groupby("target_issue_num", sort=False):
            first = grp.iloc[0]
            winning = str(first.get("winning_number", "") or "")
            actual_date = str(first.get("actual_draw_date", "") or "")
            status = str(first.get("status", "") or "")
            lines += [
                f"第{int(target)}回  予測生成={first['generated_at']}  状態={status}",
                f"予測日候補={first['target_date_candidate']}  実抽せん日={actual_date or '-'}  本数字={winning or '-'}",
                "-" * 72,
            ]
            for _, r in grp.sort_values("rank").iterrows():
                match = str(r.get("match_types", "") or "")
                prize = str(r.get("prize_details", "") or "")
                if str(r.get("status", "")) != "照合済":
                    match = "未抽せん"
                    prize = "-"
                lines.append(
                    f"{int(r['rank']):02d}. {str(r['prediction']).zfill(3)}  "
                    f"p={float(r['model_probability']):.6%}  判定={match or '-'}  払戻={prize or '-'}"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--scraper", default="scrapingnumbers3.py")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--validation", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--backtest-dir", default="backtests")
    ap.add_argument("--state-dir", default="state")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    pred_dir = Path(args.pred_dir)
    bt_dir = Path(args.backtest_dir)
    state_dir = Path(args.state_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    bt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        added = preserve_merge_scraped(csv_path.resolve(), Path(args.scraper).resolve())
        print(f"[refresh] added={added}")

    draws = load_draws(csv_path)
    history_path = pred_dir / "prediction_history.csv"
    history = read_history(history_path)
    history = resolve_history(history, draws)

    cfg = AgentConfig(validation_draws=args.validation, top_k=args.topk)
    report = run_agent(
        csv_path.resolve(),
        pred_dir.resolve(),
        (state_dir / "numbers3_agent_state.json").resolve(),
        cfg,
    )

    history, appended = append_predictions(history, report)
    history = resolve_history(history, draws)
    history.to_csv(history_path, index=False, encoding="utf-8-sig")

    append_backtest(bt_dir / "backtest_history.csv", report)
    render_latest(pred_dir / "latest_prediction.txt", report)
    render_cumulative(pred_dir / "cumulative_results.txt", history)

    run_summary = {
        "generated_at": report["generated_at"],
        "generation": report["generation"],
        "source_last_issue": report["dataset"]["last_issue"],
        "target_issue": issue_no(report["dataset"]["last_issue"]) + 1,
        "prediction_appended": appended,
        "history_rows": int(len(history)),
        "edge_status": report["validation"]["edge_status"],
    }
    (pred_dir / "last_run.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
