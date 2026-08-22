#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import numbers3_evo_agent as agent
import numbers3_champion_replay as old
from numbers3_champion import load_champion
from numbers3_v3_models import (
    build_features,
    extended_config,
    final_joint,
    fit_ml_models,
    frequency_probs,
    markov_probs,
    top_numbers,
    v3_config_id,
)

JST = ZoneInfo("Asia/Tokyo")
U_DIGIT = math.log(10.0)
U_EXACT = math.log(1000.0)
KS = (1, 3, 5, 10, 20)


def digit_nll(actual: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([-math.log(max(float(p[i, int(actual[i])]), 1e-15)) for i in range(3)]))


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "n": n,
        "exact_nll": float(df.exact_nll.mean()),
        "digit_nll": float(df.digit_nll.mean()),
        "mean_rank": float(df.actual_rank.mean()),
        "top1_rate": float(df.hit1.mean()),
        "top3_rate": float(df.hit3.mean()),
        "top5_rate": float(df.hit5.mean()),
        "top10_rate": float(df.hit10.mean()),
        "top20_rate": float(df.hit20.mean()),
        "box20_rate": float(df.box20.mean()),
        "mini20_rate": float(df.mini20.mean()),
    }


def run_replay(csv_path: Path, champion_path: Path, state_path: Path, outdir: Path,
               warmup: int = 500, refit_interval: int = 500, audit_draws: int = 1000,
               ewma_alpha: float = 0.02, force: bool = False) -> dict:
    df, nums = agent.load_history(csv_path)
    champion = load_champion(champion_path)
    cfg = extended_config(champion["config"])
    champion["config"] = cfg
    fp = old.fingerprint(df)
    cid = v3_config_id(cfg)
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    if (not force and state.get("dataset_fingerprint") == fp and state.get("champion_id") == champion["champion_id"]
            and state.get("config_id") == cid and state.get("replay_version") == "v3"
            and (outdir / "latest_rounds.csv").exists()):
        print(json.dumps({**state, "skipped": True}, ensure_ascii=False, indent=2))
        return {**state, "skipped": True}
    if len(df) <= warmup + audit_draws + 200:
        raise ValueError("履歴が不足しています")

    X = build_features(nums, cfg)
    losses = {name: U_DIGIT for name in agent.MODEL_NAMES}
    prev_weights = None
    rows = []
    train_start = max(agent.LAGS)

    for bs in range(warmup, len(df), refit_interval):
        be = min(bs + refit_interval, len(df))
        models = fit_ml_models(X.iloc[train_start:bs], nums[train_start:bs], cfg, seed=cfg["random_state"])
        ml = agent.predict_ml(models, X.iloc[bs:be])
        refit_issue = old.issue_no(df.iloc[bs - 1]["回別"])
        for j, t in enumerate(range(bs, be)):
            hist = nums[:t]
            probs = {
                "frequency": frequency_probs(hist, cfg)[None, :, :],
                "markov": markov_probs(hist, cfg)[None, :, :],
                "logreg": ml["logreg"][j:j+1],
                "extra_trees": ml["extra_trees"][j:j+1],
            }
            current = agent.derive_weights(losses, cfg["weight_temperature"])
            weights = agent.blend_weights(current, prev_weights, cfg["evolution_rate"]) if prev_weights else current
            prev_weights = weights
            base_pos = agent.ensemble_probs(probs, weights)[0]
            joint = final_joint(base_pos, hist, cfg)
            actual = old.num3(nums[t])
            ai = int(actual)
            ap = max(float(joint[ai]), 1e-15)
            rank = 1 + int(np.sum(joint > ap))
            top = [n for n, _ in top_numbers(joint, 20)]
            cube = joint.reshape(10, 10, 10)
            marg = np.stack([cube.sum(axis=(1,2)), cube.sum(axis=(0,2)), cube.sum(axis=(0,1))])
            row = {
                "target_issue": old.issue_no(df.iloc[t]["回別"]),
                "draw_date": str(pd.to_datetime(df.iloc[t]["抽せん日"]).date()),
                "actual": actual, "refit_issue": refit_issue, "top1": top[0],
                "top20": " ".join(top), "actual_rank": rank,
                "exact_nll": -math.log(ap), "digit_nll": digit_nll(nums[t], marg),
                "box20": int(any(sorted(x) == sorted(actual) for x in top)),
                "mini20": int(any(x[-2:] == actual[-2:] for x in top)),
            }
            for k in KS:
                row[f"hit{k}"] = int(actual in top[:k])
            rows.append(row)
            for name in agent.MODEL_NAMES:
                loss = digit_nll(nums[t], probs[name][0])
                losses[name] = (1.0 - ewma_alpha) * losses[name] + ewma_alpha * loss

    result = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(outdir / "latest_rounds.csv", index=False, encoding="utf-8-sig")
    full = summarize(result)
    split = max(1, len(result) - audit_draws)
    selection, audit = summarize(result.iloc[:split]), summarize(result.iloc[split:])
    now = datetime.now(JST).isoformat(timespec="seconds")
    previous_best = state.get("best_selection_exact_nll")
    if previous_best is None:
        improvement, best, improved, stagnation = 0.0, selection["exact_nll"], True, 0
    else:
        previous_best = float(previous_best)
        improvement = previous_best - selection["exact_nll"]
        improved = bool(improvement > 1e-9)
        best = min(previous_best, selection["exact_nll"])
        stagnation = 0 if improved else int(state.get("stagnation_count", 0)) + 1

    replay = {
        "generated_at": now, "dataset_fingerprint": fp, "champion_id": champion["champion_id"],
        "generation": int(champion.get("generation", 1)), "config_id": cid,
        "latest_issue": old.issue_no(df.iloc[-1]["回別"]), "evaluated_draws": len(result),
        "warmup": warmup, "refit_interval": refit_interval, "audit_draws": audit_draws,
        "selection": selection, "audit_report_only": audit, "full": full,
        "best_selection_exact_nll": float(best), "improvement_vs_previous_best": float(improvement),
        "improved_best": improved, "stagnation_count": stagnation, "replay_version": "v3", "skipped": False,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    champion["historical_replay"] = {
        "generated_at": now, "dataset_fingerprint": fp, "selection": selection,
        "audit_report_only": audit, "full": full, "stagnation_count": stagnation, "replay_version": "v3",
    }
    champion_path.write_text(json.dumps(champion, ensure_ascii=False, indent=2), encoding="utf-8")

    history_row = {
        "generated_at": now, "champion_id": champion["champion_id"], "generation": champion.get("generation", 1),
        "config_id": cid, "dataset_fingerprint": fp, "latest_issue": replay["latest_issue"],
        "evaluated_draws": len(result), "selection_draws": selection["n"], "audit_draws": audit["n"],
        "selection_exact_nll": selection["exact_nll"], "selection_improvement_vs_uniform": U_EXACT-selection["exact_nll"],
        "selection_top1_rate": selection["top1_rate"], "selection_top20_rate": selection["top20_rate"],
        "selection_mean_rank": selection["mean_rank"], "audit_exact_nll": audit["exact_nll"],
        "audit_top1_rate": audit["top1_rate"], "audit_top20_rate": audit["top20_rate"],
        "audit_mean_rank": audit["mean_rank"], "best_selection_exact_nll": best,
        "improvement_vs_previous_best": improvement, "improved_best": improved, "stagnation_count": stagnation,
    }
    old.append_history(outdir / "replay_history.csv", history_row)
    old.render_summary(outdir / "latest_accuracy.txt", champion, full, selection, audit, replay, audit_draws, refit_interval)
    print(json.dumps(replay, ensure_ascii=False, indent=2))
    return replay


def main():
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
               a.warmup, a.refit_interval, a.audit_draws, force=a.force)


if __name__ == "__main__":
    main()
