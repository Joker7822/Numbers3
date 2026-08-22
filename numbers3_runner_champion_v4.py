#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import numbers3_evo_agent as agent
import numbers3_runner as runner
from numbers3_champion import load_champion
from numbers3_champion_pool import load_pool
from numbers3_live_scoreboard import update_scoreboard
from numbers3_runner_champion_v3 import append_live_archive
from numbers3_v4_models import (
    U_EXACT,
    build_features,
    extended_config,
    final_joint,
    fit_ml_models,
    frequency_probs,
    make_next_features,
    markov_probs,
    metrics_from_joint_rows,
    top_numbers,
    v4_config_id,
)

JST = ZoneInfo("Asia/Tokyo")


def _feature_matrix(nums: np.ndarray, cfg: dict, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = str(cfg.get("feature_profile", "base"))
    if key not in cache:
        cache[key] = build_features(nums, cfg)
    return cache[key]


def _member_predictions(nums: np.ndarray, split: int, cal_n: int, cfg: dict, feature_cache: dict[str, pd.DataFrame]):
    cfg = extended_config(cfg)
    X = _feature_matrix(nums, cfg, feature_cache)
    start = max(agent.LAGS)
    models = fit_ml_models(X.iloc[start:split], nums[start:split], cfg, seed=cfg["random_state"])
    ml = agent.predict_ml(models, X.iloc[split:])
    val_n = len(nums) - split
    freq = np.zeros((val_n, 3, 10), dtype=float)
    markov = np.zeros_like(freq)
    for j, t in enumerate(range(split, len(nums))):
        hist = nums[:t]
        freq[j] = frequency_probs(hist, cfg)
        markov[j] = markov_probs(hist, cfg)
    components = {"frequency": freq, "markov": markov, **ml}
    y_cal = nums[split:split+cal_n]
    losses = {name: agent.mean_position_logloss(y_cal, p[:cal_n]) for name, p in components.items()}
    weights = agent.derive_weights(losses, cfg["weight_temperature"])

    joints = []
    for j, t in enumerate(range(split + cal_n, len(nums)), start=cal_n):
        one = {name: p[j:j+1] for name, p in components.items()}
        base_pos = agent.ensemble_probs(one, weights)[0]
        joints.append(final_joint(base_pos, nums[:t], cfg))

    full_models = fit_ml_models(X.iloc[start:], nums[start:], cfg, seed=cfg["random_state"])
    X_next = make_next_features(nums, cfg)
    next_components = {
        "frequency": frequency_probs(nums, cfg)[None, :, :],
        "markov": markov_probs(nums, cfg)[None, :, :],
        **agent.predict_ml(full_models, X_next),
    }
    base_next = agent.ensemble_probs(next_components, weights)[0]
    next_joint = final_joint(base_next, nums, cfg)
    return np.asarray(joints), next_joint, weights, losses


def make_run_agent(champion: dict, pool: dict):
    members = pool.get("members") or []
    if not members:
        members = [{
            "champion_id": champion["champion_id"], "generation": champion.get("generation", 1),
            "config_id": v4_config_id(champion["config"]), "config": extended_config(champion["config"]), "weight": 1.0,
        }]
    pool_weights = np.asarray([float(m.get("weight", 0.0)) for m in members], dtype=float)
    if pool_weights.sum() <= 0:
        pool_weights = np.ones(len(members), dtype=float)
    pool_weights /= pool_weights.sum()

    def run_agent_v4(csv_path: Path, outdir: Path, state_path: Path, runtime_cfg) -> dict:
        outdir.mkdir(parents=True, exist_ok=True)
        df, nums = agent.load_history(csv_path)
        validation_draws = min(int(runtime_cfg.validation_draws), max(300, len(df) // 4))
        split = len(df) - validation_draws
        start = max(agent.LAGS)
        if split <= start + 200:
            raise ValueError("validation training data is insufficient")
        cal_n = min(300, max(100, validation_draws // 3))
        score_n = validation_draws - cal_n

        feature_cache: dict[str, pd.DataFrame] = {}
        member_joints, member_next, member_component_weights, member_losses = [], [], [], []
        for m in members:
            cfg = extended_config(m["config"])
            j, nxt, w, losses = _member_predictions(nums, split, cal_n, cfg, feature_cache)
            member_joints.append(j)
            member_next.append(nxt)
            member_component_weights.append(w)
            member_losses.append(losses)

        joints = np.tensordot(pool_weights, np.stack(member_joints), axes=(0, 0))
        next_joint = np.tensordot(pool_weights, np.stack(member_next), axes=(0, 0))
        next_joint = np.clip(next_joint, 1e-15, None)
        next_joint /= next_joint.sum()
        metrics = metrics_from_joint_rows(nums[split+cal_n:], joints)
        metrics.pop("score_exact_nlls", None)
        ranking = top_numbers(next_joint, int(runtime_cfg.top_k))

        aggregate_weights = {name: 0.0 for name in agent.MODEL_NAMES}
        aggregate_losses = {name: 0.0 for name in agent.MODEL_NAMES}
        for pw, w, losses in zip(pool_weights, member_component_weights, member_losses):
            for name in aggregate_weights:
                aggregate_weights[name] += float(pw) * float(w.get(name, 0.0))
                aggregate_losses[name] += float(pw) * float(losses.get(name, math.log(10.0)))

        rank_df = pd.DataFrame([
            {"rank": i+1, "number": n, "model_probability": p, "uniform_probability": 0.001,
             "relative_to_uniform": p/0.001}
            for i, (n, p) in enumerate(ranking)
        ])
        rank_df.to_csv(outdir / "numbers3_prediction_latest.csv", index=False, encoding="utf-8-sig")

        last_date = pd.to_datetime(df.iloc[-1]["抽せん日"])
        next_date = agent.next_weekday(last_date)
        fp = agent.dataset_fingerprint(df)
        exact_nll = float(metrics["exact_nll"])
        digit_nll = float(metrics["digit_nll"])
        edge_digit = math.log(10.0) - digit_nll
        edge_exact = U_EXACT - exact_nll
        pool_id = str(pool.get("pool_id") or f"pool-{v4_config_id(champion['config'])}")
        report = {
            "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
            "generation": int(champion.get("generation", 1)),
            "champion_generation": int(champion.get("generation", 1)),
            "champion_id": champion["champion_id"],
            "config_id": pool_id,
            "primary_config_id": v4_config_id(champion["config"]),
            "dataset_fingerprint": fp,
            "dataset": {
                "rows": int(len(df)), "first_draw_date": str(pd.to_datetime(df.iloc[0]["抽せん日"]).date()),
                "last_draw_date": str(last_date.date()), "last_issue": str(df.iloc[-1]["回別"]),
                "last_number": "".join(map(str, nums[-1])), "model_next_weekday_candidate": str(next_date.date()),
            },
            "validation": {
                "draws": score_n, "calibration_draws": cal_n,
                "period_start": str(pd.to_datetime(df.iloc[split+cal_n]["抽せん日"]).date()),
                "period_end": str(last_date.date()), "model_logloss": aggregate_losses,
                "ensemble_logloss": digit_nll, "uniform_logloss": math.log(10.0),
                "logloss_improvement_vs_uniform": edge_digit,
                "exact_nll": exact_nll, "uniform_exact_nll": U_EXACT,
                "exact_nll_improvement_vs_uniform": edge_exact,
                "top1_straight_hit_rate": float(metrics["top1_rate"]),
                "top20_straight_hit_rate": float(metrics["top20_rate"]),
                "uniform_expected_top1_rate": 0.001, "uniform_expected_top20_rate": 0.02,
                "edge_status": "positive" if edge_digit > 0 and edge_exact > 0 else "not_confirmed",
            },
            "model_weights": aggregate_weights,
            "v4_config": extended_config(champion["config"]),
            "champion_pool": {
                "pool_id": pool_id, "size": len(members),
                "members": [
                    {"champion_id": m.get("champion_id"), "config_id": m.get("config_id") or v4_config_id(m["config"]),
                     "weight": float(w), "dev_exact_nll": m.get("dev_exact_nll")}
                    for m, w in zip(members, pool_weights)
                ],
            },
            "top_predictions": [{"rank": i+1, "number": n, "probability": p} for i, (n, p) in enumerate(ranking)],
            "notes": [
                "Prediction is a weighted Champion Pool ensemble; promotion governance still uses strict single-Champion comparisons.",
                "Each member is power-calibrated after uniform shrinkage and optional autoregressive mixing.",
                "Live archive stores the full 1000-number distribution before the draw.",
            ],
        }
        (outdir / "numbers3_report_latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "generation": report["generation"], "champion_id": report["champion_id"],
            "config_id": report["config_id"], "primary_config_id": report["primary_config_id"],
            "dataset_fingerprint": fp, "last_draw_date": str(last_date.date()),
            "model_weights": aggregate_weights, "last_validation": report["validation"],
            "last_top_predictions": report["top_predictions"], "champion_pool": report["champion_pool"],
            "updated_at": report["generated_at"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        append_live_archive(outdir / "live_forecast_archive.csv", report, next_joint)
        return report

    return run_agent_v4


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--champion", default="state/evolution_champion.json")
    p.add_argument("--pool", default="state/champion_pool.json")
    p.add_argument("--csv", default="data/numbers3.csv")
    p.add_argument("--pred-dir", default="predictions")
    known, rest = p.parse_known_args()
    champion = load_champion(Path(known.champion))
    champion["config"] = extended_config(champion.get("config"))
    pool = load_pool(Path(known.pool), champion)

    runner.run_agent = make_run_agent(champion, pool)
    original_render = runner.render_latest

    def render_v4(path: Path, report: dict):
        original_render(path, report)
        text = path.read_text(encoding="utf-8")
        marker = f"世代: {report['generation']}"
        replacement = (
            f"Champion世代: {report['champion_generation']}\n"
            f"Primary Champion ID: {report['champion_id']}\n"
            f"Champion Pool ID: {report['config_id']}\n"
            f"Pool size: {report['champion_pool']['size']}"
        )
        text = text.replace(marker, replacement)
        v = report["validation"]
        cfg = report["v4_config"]
        text += (
            "\nV4 joint評価\n" + "-"*64 + "\n"
            f"Exact NLL: {v['exact_nll']:.8f}\n"
            f"Uniform Exact NLL: {v['uniform_exact_nll']:.8f}\n"
            f"差(一様-model): {v['exact_nll_improvement_vs_uniform']:+.8f}\n"
            f"primary uniform_blend={cfg['uniform_blend']} / power_beta={cfg['power_beta']} / "
            f"autoregressive_mix={cfg['autoregressive_mix']} / feature_profile={cfg['feature_profile']}\n"
            "Champion Pool:\n"
        )
        for m in report["champion_pool"]["members"]:
            text += f"  {m['champion_id']} weight={m['weight']:.4%} dev_nll={m.get('dev_exact_nll')}\n"
        path.write_text(text, encoding="utf-8")

    runner.render_latest = render_v4
    forwarded = ["--csv", known.csv, "--pred-dir", known.pred_dir] + rest
    sys.argv = [sys.argv[0]] + forwarded
    runner.main()

    update_scoreboard(
        Path(known.csv), Path(known.pred_dir) / "live_forecast_archive.csv",
        Path(known.pred_dir) / "live_scoreboard.csv", Path(known.pred_dir) / "live_scoreboard.txt",
    )


if __name__ == "__main__":
    main()
