#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay all historical NUMBERS3 draws with the current Champion, causally.

The latest Champion configuration is evaluated using only information available before
 each target draw. The final audit tail is report-only and excluded from feedback used
 to steer future experiment exploration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import numbers3_evo_agent as agent
from numbers3_champion import (
    config_id,
    fit_ml_models,
    frequency_probs,
    load_champion,
    markov_probs,
    normalize_config,
)

JST = ZoneInfo("Asia/Tokyo")
KS = (1, 3, 5, 10, 20)
U_EXACT = math.log(1000.0)
U_DIGIT = math.log(10.0)
HISTORY_COLUMNS = [
    "generated_at", "champion_id", "generation", "config_id", "dataset_fingerprint",
    "latest_issue", "evaluated_draws", "selection_draws", "audit_draws",
    "selection_exact_nll", "selection_improvement_vs_uniform", "selection_top1_rate",
    "selection_top20_rate", "selection_mean_rank", "audit_exact_nll", "audit_top1_rate",
    "audit_top20_rate", "audit_mean_rank", "best_selection_exact_nll",
    "improvement_vs_previous_best", "improved_best", "stagnation_count",
]


def issue_no(value) -> int:
    m = re.search(r"(\d+)", str(value))
    if not m:
        raise ValueError(f"回号を解釈できません: {value!r}")
    return int(m.group(1))


def fingerprint(df: pd.DataFrame) -> str:
    cols = [c for c in ("抽せん日", "本数字", "回別") if c in df.columns]
    payload = df[cols].fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def num3(row: np.ndarray) -> str:
    return "".join(str(int(x)) for x in row)


def digit_nll(actual: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([-math.log(max(float(p[pos, int(actual[pos])]), 1e-15)) for pos in range(3)]))


def joint_probs(p: np.ndarray) -> np.ndarray:
    p = agent.normalize_probs(np.asarray(p, dtype=float))
    return (p[0, :, None, None] * p[1, None, :, None] * p[2, None, None, :]).reshape(-1)


def summarize(frame: pd.DataFrame) -> dict:
    n = len(frame)
    if n == 0:
        return {"n": 0, "exact_nll": float("nan"), "digit_nll": float("nan"),
                "mean_rank": float("nan"), "top1_rate": 0.0, "top20_rate": 0.0}
    return {
        "n": n,
        "exact_nll": float(frame["exact_nll"].mean()),
        "digit_nll": float(frame["digit_nll"].mean()),
        "mean_rank": float(frame["actual_rank"].mean()),
        "top1_rate": float(frame["hit1"].mean()),
        "top3_rate": float(frame["hit3"].mean()),
        "top5_rate": float(frame["hit5"].mean()),
        "top10_rate": float(frame["hit10"].mean()),
        "top20_rate": float(frame["hit20"].mean()),
        "box20_rate": float(frame["box20"].mean()),
        "mini20_rate": float(frame["mini20"].mean()),
    }


def append_history(path: Path, row: dict, max_rows: int = 20000) -> None:
    if path.exists():
        hist = pd.read_csv(path)
    else:
        hist = pd.DataFrame(columns=HISTORY_COLUMNS)
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    for col in HISTORY_COLUMNS:
        if col not in hist.columns:
            hist[col] = ""
    hist = hist.tail(max_rows).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist[HISTORY_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")


def render_summary(path: Path, champion: dict, full: dict, selection: dict, audit: dict,
                   state: dict, audit_draws: int, refit_interval: int) -> None:
    lines = [
        "NUMBERS3 最新Champion 全過去回リプレイ", "=" * 92,
        f"生成日時: {state['generated_at']}",
        f"Champion: {champion['champion_id']} / generation={champion['generation']}",
        f"設定ID: {config_id(champion['config'])}",
        f"対象最新回: 第{state['latest_issue']}回",
        f"評価回数: {full['n']:,}回 / ML再学習間隔: {refit_interval}回",
        "検証方式: 第N回は第N-1回以前だけを使用する causal walk-forward",
        "", "選択用履歴（最終holdoutを除外。次の探索フィードバックに使用）", "-" * 92,
        f"件数: {selection['n']:,}回",
        f"Exact NLL: {selection['exact_nll']:.8f} / 一様: {U_EXACT:.8f} / 改善: {U_EXACT-selection['exact_nll']:+.8f}",
        f"1桁 log loss: {selection['digit_nll']:.8f} / 一様: {U_DIGIT:.8f}",
        f"平均順位: {selection['mean_rank']:.2f}/1000",
        f"Top1: {selection['top1_rate']:.4%} / Top5: {selection['top5_rate']:.4%} / Top20: {selection['top20_rate']:.4%}",
        f"BOX20: {selection['box20_rate']:.4%} / ミニ20: {selection['mini20_rate']:.4%}",
        f"過去ベスト selection NLL: {state['best_selection_exact_nll']:.8f}",
        f"過去ベストとの差: {state['improvement_vs_previous_best']:+.8f}",
        f"ベスト更新: {'YES' if state['improved_best'] else 'NO'} / 停滞カウント: {state['stagnation_count']}",
        "", f"Final audit holdout（最後の{audit_draws}回・報告専用）", "-" * 92,
        f"件数: {audit['n']:,}回",
        f"Exact NLL: {audit['exact_nll']:.8f} / 一様: {U_EXACT:.8f} / 改善: {U_EXACT-audit['exact_nll']:+.8f}",
        f"平均順位: {audit['mean_rank']:.2f}/1000",
        f"Top1: {audit['top1_rate']:.4%} / Top20: {audit['top20_rate']:.4%}",
        "", "全評価期間", "-" * 92,
        f"Exact NLL: {full['exact_nll']:.8f} / 平均順位: {full['mean_rank']:.2f}/1000",
        f"Top1: {full['top1_rate']:.4%} / Top20: {full['top20_rate']:.4%}",
        "", "重要:",
        "・Final auditはChampion昇格判定や探索方向の決定には使用しません。",
        "・同じ過去データを繰り返し最適化すると過学習し得るため、未知回の精度上昇は保証されません。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_replay(csv_path: Path, champion_path: Path, state_path: Path, outdir: Path,
               warmup: int = 500, refit_interval: int = 500, audit_draws: int = 1000,
               ewma_alpha: float = 0.02, force: bool = False) -> dict:
    df, nums = agent.load_history(csv_path)
    champion = load_champion(champion_path)
    cfg = normalize_config(champion["config"])
    fp = fingerprint(df)
    cid = config_id(cfg)
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    if not force and state.get("dataset_fingerprint") == fp and state.get("champion_id") == champion["champion_id"] \
            and state.get("config_id") == cid and (outdir / "latest_rounds.csv").exists():
        print(json.dumps({**state, "skipped": True}, ensure_ascii=False, indent=2))
        return {**state, "skipped": True}

    if len(df) <= warmup + audit_draws + 200:
        raise ValueError("履歴が不足しています")

    X = agent.build_features(nums)
    losses = {name: U_DIGIT for name in agent.MODEL_NAMES}
    prev_weights = None
    rows = []

    for bs in range(warmup, len(df), refit_interval):
        be = min(bs + refit_interval, len(df))
        train_start = max(agent.LAGS)
        models = fit_ml_models(agent, X.iloc[train_start:bs], nums[train_start:bs], cfg, seed=cfg["random_state"])
        ml = agent.predict_ml(models, X.iloc[bs:be])
        refit_issue = issue_no(df.iloc[bs - 1]["回別"])
        for j, t in enumerate(range(bs, be)):
            hist = nums[:t]
            probs = {
                "frequency": frequency_probs(agent, hist, cfg)[None, :, :],
                "markov": markov_probs(agent, hist, cfg)[None, :, :],
                "logreg": ml["logreg"][j:j+1],
                "extra_trees": ml["extra_trees"][j:j+1],
            }
            current_weights = agent.derive_weights(losses, cfg["weight_temperature"])
            weights = agent.blend_weights(current_weights, prev_weights, cfg["evolution_rate"]) if prev_weights else current_weights
            prev_weights = weights
            ens = agent.ensemble_probs(probs, weights)[0]
            jp = joint_probs(ens)
            actual = num3(nums[t])
            actual_idx = int(actual)
            ap = max(float(jp[actual_idx]), 1e-15)
            order = np.argsort(jp)[::-1]
            rank = int(np.where(order == actual_idx)[0][0]) + 1
            top20 = [f"{int(z):03d}" for z in order[:20]]
            row = {
                "target_issue": issue_no(df.iloc[t]["回別"]),
                "draw_date": str(pd.to_datetime(df.iloc[t]["抽せん日"]).date()),
                "actual": actual,
                "refit_issue": refit_issue,
                "top1": top20[0],
                "top20": " ".join(top20),
                "actual_rank": rank,
                "exact_nll": -math.log(ap),
                "digit_nll": digit_nll(nums[t], ens),
                "box20": int(any(sorted(x) == sorted(actual) for x in top20)),
                "mini20": int(any(x[-2:] == actual[-2:] for x in top20)),
            }
            for k in KS:
                row[f"hit{k}"] = int(actual in top20[:k])
            rows.append(row)
            for name in agent.MODEL_NAMES:
                loss = digit_nll(nums[t], probs[name][0])
                losses[name] = (1.0 - ewma_alpha) * losses[name] + ewma_alpha * loss

    result = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(outdir / "latest_rounds.csv", index=False, encoding="utf-8-sig")

    full = summarize(result)
    split = max(1, len(result) - audit_draws)
    selection = summarize(result.iloc[:split])
    audit = summarize(result.iloc[split:])
    now = datetime.now(JST).isoformat(timespec="seconds")
    previous_best = state.get("best_selection_exact_nll")
    if previous_best is None:
        improvement = 0.0
        best = selection["exact_nll"]
        improved_best = True
        stagnation = 0
    else:
        previous_best = float(previous_best)
        improvement = previous_best - selection["exact_nll"]
        improved_best = bool(improvement > 1e-9)
        best = min(previous_best, selection["exact_nll"])
        stagnation = 0 if improved_best else int(state.get("stagnation_count", 0)) + 1

    replay = {
        "generated_at": now,
        "dataset_fingerprint": fp,
        "champion_id": champion["champion_id"],
        "generation": int(champion.get("generation", 1)),
        "config_id": cid,
        "latest_issue": issue_no(df.iloc[-1]["回別"]),
        "evaluated_draws": len(result),
        "warmup": warmup,
        "refit_interval": refit_interval,
        "audit_draws": audit_draws,
        "selection": selection,
        "audit_report_only": audit,
        "full": full,
        "best_selection_exact_nll": float(best),
        "improvement_vs_previous_best": float(improvement),
        "improved_best": improved_best,
        "stagnation_count": stagnation,
        "skipped": False,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")

    champion["historical_replay"] = {
        "generated_at": now,
        "dataset_fingerprint": fp,
        "selection": selection,
        "audit_report_only": audit,
        "full": full,
        "stagnation_count": stagnation,
    }
    champion_path.write_text(json.dumps(champion, ensure_ascii=False, indent=2), encoding="utf-8")

    history_row = {
        "generated_at": now,
        "champion_id": champion["champion_id"],
        "generation": champion.get("generation", 1),
        "config_id": cid,
        "dataset_fingerprint": fp,
        "latest_issue": replay["latest_issue"],
        "evaluated_draws": len(result),
        "selection_draws": selection["n"],
        "audit_draws": audit["n"],
        "selection_exact_nll": selection["exact_nll"],
        "selection_improvement_vs_uniform": U_EXACT-selection["exact_nll"],
        "selection_top1_rate": selection["top1_rate"],
        "selection_top20_rate": selection["top20_rate"],
        "selection_mean_rank": selection["mean_rank"],
        "audit_exact_nll": audit["exact_nll"],
        "audit_top1_rate": audit["top1_rate"],
        "audit_top20_rate": audit["top20_rate"],
        "audit_mean_rank": audit["mean_rank"],
        "best_selection_exact_nll": best,
        "improvement_vs_previous_best": improvement,
        "improved_best": improved_best,
        "stagnation_count": stagnation,
    }
    append_history(outdir / "replay_history.csv", history_row)
    render_summary(outdir / "latest_accuracy.txt", champion, full, selection, audit, replay, audit_draws, refit_interval)
    print(json.dumps(replay, ensure_ascii=False, indent=2))
    return replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--champion", default="state/evolution_champion.json")
    ap.add_argument("--state", default="state/champion_replay_state.json")
    ap.add_argument("--output-dir", default="backtests/champion_replay")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refit-interval", type=int, default=500)
    ap.add_argument("--audit-draws", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_replay(Path(a.csv), Path(a.champion), Path(a.state), Path(a.output_dir),
               warmup=a.warmup, refit_interval=a.refit_interval,
               audit_draws=a.audit_draws, force=a.force)


if __name__ == "__main__":
    main()
