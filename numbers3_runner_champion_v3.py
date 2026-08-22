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
from numbers3_live_scoreboard import update_scoreboard
from numbers3_v3_models import (
    U_EXACT,
    build_features,
    extended_config,
    final_joint,
    fit_ml_models,
    frequency_probs,
    joint_marginals,
    make_next_features,
    markov_probs,
    top_numbers,
    v3_config_id,
)

JST = ZoneInfo("Asia/Tokyo")


def append_live_archive(path: Path, report: dict, joint: np.ndarray) -> None:
    d = report["dataset"]
    target = runner.issue_no(d["last_issue"]) + 1
    row = {
        "generated_at": report["generated_at"],
        "source_last_issue": runner.issue_no(d["last_issue"]),
        "target_issue": target,
        "target_date_candidate": d["model_next_weekday_candidate"],
        "champion_id": report["champion_id"],
        "generation": report["champion_generation"],
        "config_id": report["config_id"],
        "probabilities_json": json.dumps([round(float(x), 12) for x in joint], separators=(",", ":")),
    }
    cols = list(row)
    if path.exists():
        old = pd.read_csv(path, dtype={"probabilities_json": str})
    else:
        old = pd.DataFrame(columns=cols)
    if not old.empty:
        same = (
            pd.to_numeric(old["target_issue"], errors="coerce").eq(target)
            & old["champion_id"].astype(str).eq(report["champion_id"])
            & old["config_id"].astype(str).eq(report["config_id"])
        )
        if same.any():
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).tail(2000).to_csv(path, index=False, encoding="utf-8-sig")


def make_run_agent(champion: dict):
    cfg = extended_config(champion["config"])

    def run_agent_v3(csv_path: Path, outdir: Path, state_path: Path, runtime_cfg) -> dict:
        outdir.mkdir(parents=True, exist_ok=True)
        df, nums = agent.load_history(csv_path)
        X = build_features(nums, cfg)
        start = max(agent.LAGS)
        validation_draws = min(int(runtime_cfg.validation_draws), max(300, len(df) // 4))
        split = len(df) - validation_draws
        if split <= start + 200:
            raise ValueError("validation training data is insufficient")

        models = fit_ml_models(X.iloc[start:split], nums[start:split], cfg, seed=cfg["random_state"])
        ml = agent.predict_ml(models, X.iloc[split:])
        freq = np.zeros((validation_draws, 3, 10), dtype=float)
        markov = np.zeros_like(freq)
        for j, t in enumerate(range(split, len(df))):
            hist = nums[:t]
            freq[j] = frequency_probs(hist, cfg)
            markov[j] = markov_probs(hist, cfg)
        components = {"frequency": freq, "markov": markov, **ml}

        cal_n = min(300, max(100, validation_draws // 3))
        y_cal = nums[split:split+cal_n]
        losses = {name: agent.mean_position_logloss(y_cal, p[:cal_n]) for name, p in components.items()}
        weights = agent.derive_weights(losses, cfg["weight_temperature"])

        exact_losses, digit_losses = [], []
        top1 = top20 = 0
        for j, t in enumerate(range(split + cal_n, len(df)), start=cal_n):
            one = {name: p[j:j+1] for name, p in components.items()}
            base_pos = agent.ensemble_probs(one, weights)[0]
            joint = final_joint(base_pos, nums[:t], cfg)
            ai = int(nums[t, 0]) * 100 + int(nums[t, 1]) * 10 + int(nums[t, 2])
            exact_losses.append(-math.log(max(float(joint[ai]), 1e-15)))
            marg = joint_marginals(joint)
            digit_losses.append(np.mean([-math.log(max(float(marg[p, int(nums[t, p])]), 1e-15)) for p in range(3)]))
            order20 = np.argpartition(joint, -20)[-20:]
            top20 += int(ai in order20)
            top1 += int(ai == int(np.argmax(joint)))
        score_n = validation_draws - cal_n
        exact_nll = float(np.mean(exact_losses))
        digit_nll = float(np.mean(digit_losses))

        full_models = fit_ml_models(X.iloc[start:], nums[start:], cfg, seed=cfg["random_state"])
        X_next = make_next_features(nums, cfg)
        next_components = {
            "frequency": frequency_probs(nums, cfg)[None, :, :],
            "markov": markov_probs(nums, cfg)[None, :, :],
            **agent.predict_ml(full_models, X_next),
        }
        base_next = agent.ensemble_probs(next_components, weights)[0]
        next_joint = final_joint(base_next, nums, cfg)
        ranking = top_numbers(next_joint, int(runtime_cfg.top_k))

        rank_df = pd.DataFrame([
            {"rank": i+1, "number": n, "model_probability": p, "uniform_probability": 0.001,
             "relative_to_uniform": p/0.001}
            for i, (n, p) in enumerate(ranking)
        ])
        rank_df.to_csv(outdir / "numbers3_prediction_latest.csv", index=False, encoding="utf-8-sig")

        last_date = pd.to_datetime(df.iloc[-1]["抽せん日"])
        next_date = agent.next_weekday(last_date)
        fp = agent.dataset_fingerprint(df)
        edge_digit = math.log(10.0) - digit_nll
        edge_exact = U_EXACT - exact_nll
        report = {
            "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
            "generation": int(champion.get("generation", 1)),
            "champion_generation": int(champion.get("generation", 1)),
            "champion_id": champion["champion_id"],
            "config_id": v3_config_id(cfg),
            "dataset_fingerprint": fp,
            "dataset": {
                "rows": int(len(df)), "first_draw_date": str(pd.to_datetime(df.iloc[0]["抽せん日"]).date()),
                "last_draw_date": str(last_date.date()), "last_issue": str(df.iloc[-1]["回別"]),
                "last_number": "".join(map(str, nums[-1])), "model_next_weekday_candidate": str(next_date.date()),
            },
            "validation": {
                "draws": score_n,
                "calibration_draws": cal_n,
                "period_start": str(pd.to_datetime(df.iloc[split+cal_n]["抽せん日"]).date()),
                "period_end": str(last_date.date()),
                "model_logloss": losses,
                "ensemble_logloss": digit_nll,
                "uniform_logloss": math.log(10.0),
                "logloss_improvement_vs_uniform": edge_digit,
                "exact_nll": exact_nll,
                "uniform_exact_nll": U_EXACT,
                "exact_nll_improvement_vs_uniform": edge_exact,
                "top1_straight_hit_rate": top1 / score_n,
                "top20_straight_hit_rate": top20 / score_n,
                "uniform_expected_top1_rate": 0.001,
                "uniform_expected_top20_rate": 0.02,
                "edge_status": "positive" if edge_digit > 0 and edge_exact > 0 else "not_confirmed",
            },
            "model_weights": weights,
            "v3_config": cfg,
            "top_predictions": [{"rank": i+1, "number": n, "probability": p} for i, (n, p) in enumerate(ranking)],
            "notes": [
                "Champion generation is the model generation; data-generation counter is no longer shown as Champion generation.",
                "Validation uses a calibration prefix followed by untouched scoring draws.",
                "Joint prediction supports uniform shrinkage and an autoregressive within-number component.",
            ],
        }
        (outdir / "numbers3_report_latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "generation": report["generation"], "champion_id": report["champion_id"],
            "config_id": report["config_id"], "dataset_fingerprint": fp,
            "last_draw_date": str(last_date.date()), "model_weights": weights,
            "last_validation": report["validation"], "last_top_predictions": report["top_predictions"],
            "updated_at": report["generated_at"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        append_live_archive(outdir / "live_forecast_archive.csv", report, next_joint)
        return report

    return run_agent_v3


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--champion", default="state/evolution_champion.json")
    p.add_argument("--csv", default="data/numbers3.csv")
    p.add_argument("--pred-dir", default="predictions")
    known, rest = p.parse_known_args()
    champion = load_champion(Path(known.champion))
    champion["config"] = extended_config(champion.get("config"))

    runner.run_agent = make_run_agent(champion)
    original_render = runner.render_latest

    def render_v3(path: Path, report: dict):
        original_render(path, report)
        text = path.read_text(encoding="utf-8")
        marker = f"世代: {report['generation']}"
        replacement = (
            f"Champion世代: {report['champion_generation']}\n"
            f"Champion ID: {report['champion_id']}\n"
            f"設定ID: {report['config_id']}"
        )
        text = text.replace(marker, replacement)
        v = report["validation"]
        text += (
            "\nV3 joint評価\n" + "-"*64 + "\n"
            f"Exact NLL: {v['exact_nll']:.8f}\n"
            f"Uniform Exact NLL: {v['uniform_exact_nll']:.8f}\n"
            f"差(一様-model): {v['exact_nll_improvement_vs_uniform']:+.8f}\n"
            f"uniform_blend={report['v3_config']['uniform_blend']} / "
            f"autoregressive_mix={report['v3_config']['autoregressive_mix']} / "
            f"feature_profile={report['v3_config']['feature_profile']}\n"
        )
        path.write_text(text, encoding="utf-8")

    runner.render_latest = render_v3
    forwarded = ["--csv", known.csv, "--pred-dir", known.pred_dir] + rest
    sys.argv = [sys.argv[0]] + forwarded
    runner.main()

    update_scoreboard(
        Path(known.csv), Path(known.pred_dir) / "live_forecast_archive.csv",
        Path(known.pred_dir) / "live_scoreboard.csv", Path(known.pred_dir) / "live_scoreboard.txt",
    )


if __name__ == "__main__":
    main()
