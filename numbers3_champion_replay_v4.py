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
from numbers3_champion_pool import load_pool
from numbers3_v4_models import (
    U_EXACT,
    build_features,
    extended_config,
    final_joint,
    fit_ml_models,
    frequency_probs,
    markov_probs,
    top_numbers,
)

JST = ZoneInfo("Asia/Tokyo")
U_DIGIT = math.log(10.0)
KS = (1, 3, 5, 10, 20)
REPLAY_VERSION = "v4-champion-pool"


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


def render_summary(path: Path, champion: dict, pool: dict, full: dict, selection: dict, audit: dict,
                   state: dict, audit_draws: int, refit_interval: int) -> None:
    lines = [
        "NUMBERS3 Champion Pool v4 全過去回リプレイ", "=" * 96,
        f"生成日時: {state['generated_at']}",
        f"Primary Champion: {champion['champion_id']} / generation={champion['generation']}",
        f"Champion Pool: {pool['pool_id']} / members={len(pool.get('members') or [])}",
        f"対象最新回: 第{state['latest_issue']}回",
        f"評価回数: {full['n']:,}回 / ML再学習間隔: {refit_interval}回",
        "検証方式: 第N回は第N-1回以前だけを使用する causal walk-forward", "",
        "Pool members", "-" * 96,
    ]
    for m in pool.get("members") or []:
        lines.append(
            f"{m['champion_id']} cfg={m['config_id']} weight={float(m.get('weight',0)):.4%} "
            f"dev_nll={m.get('dev_exact_nll')}"
        )
    lines += [
        "", "選択用履歴（最終holdoutを除外）", "-" * 96,
        f"件数: {selection['n']:,}回",
        f"Exact NLL: {selection['exact_nll']:.8f} / 一様: {U_EXACT:.8f} / 改善: {U_EXACT-selection['exact_nll']:+.8f}",
        f"平均順位: {selection['mean_rank']:.2f}/1000",
        f"Top1: {selection['top1_rate']:.4%} / Top20: {selection['top20_rate']:.4%}",
        f"過去ベスト selection NLL: {state['best_selection_exact_nll']:.8f}",
        f"過去ベストとの差: {state['improvement_vs_previous_best']:+.8f}",
        f"ベスト更新: {'YES' if state['improved_best'] else 'NO'}", "",
        f"Final audit holdout（最後の{audit_draws}回・報告専用）", "-" * 96,
        f"Exact NLL: {audit['exact_nll']:.8f} / 一様: {U_EXACT:.8f} / 改善: {U_EXACT-audit['exact_nll']:+.8f}",
        f"平均順位: {audit['mean_rank']:.2f}/1000",
        f"Top1: {audit['top1_rate']:.4%} / Top20: {audit['top20_rate']:.4%}", "",
        "全評価期間", "-" * 96,
        f"Exact NLL: {full['exact_nll']:.8f} / 平均順位: {full['mean_rank']:.2f}/1000",
        f"Top1: {full['top1_rate']:.4%} / Top20: {full['top20_rate']:.4%}", "",
        "重要: final auditは探索・昇格・Pool重み決定には使用しません。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_replay(csv_path: Path, champion_path: Path, pool_path: Path, state_path: Path, outdir: Path,
               warmup: int = 500, refit_interval: int = 500, audit_draws: int = 1000,
               ewma_alpha: float = 0.02, force: bool = False) -> dict:
    df, nums = agent.load_history(csv_path)
    champion = load_champion(champion_path)
    champion["config"] = extended_config(champion["config"])
    pool = load_pool(pool_path, champion)
    members = pool.get("members") or []
    if not members:
        raise RuntimeError("Champion Pool is empty")
    weights_pool = np.asarray([float(m.get("weight", 0.0)) for m in members], dtype=float)
    if weights_pool.sum() <= 0:
        weights_pool = np.ones(len(members), dtype=float)
    weights_pool /= weights_pool.sum()

    fp = old.fingerprint(df)
    pid = pool["pool_id"]
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    if (not force and state.get("dataset_fingerprint") == fp and state.get("champion_pool_id") == pid
            and state.get("replay_version") == REPLAY_VERSION and (outdir / "latest_rounds.csv").exists()):
        print(json.dumps({**state, "skipped": True}, ensure_ascii=False, indent=2))
        return {**state, "skipped": True}
    if len(df) <= warmup + audit_draws + 200:
        raise ValueError("履歴が不足しています")

    feature_cache: dict[str, pd.DataFrame] = {}
    member_X = []
    member_cfg = []
    for m in members:
        cfg = extended_config(m["config"])
        member_cfg.append(cfg)
        profile = str(cfg.get("feature_profile", "base"))
        if profile not in feature_cache:
            feature_cache[profile] = build_features(nums, cfg)
        member_X.append(feature_cache[profile])

    member_losses = [{name: U_DIGIT for name in agent.MODEL_NAMES} for _ in members]
    member_prev_weights: list[dict | None] = [None for _ in members]
    rows = []
    train_start = max(agent.LAGS)

    for bs in range(warmup, len(df), refit_interval):
        be = min(bs + refit_interval, len(df))
        block_ml = []
        for cfg, X in zip(member_cfg, member_X):
            models = fit_ml_models(X.iloc[train_start:bs], nums[train_start:bs], cfg, seed=cfg["random_state"])
            block_ml.append(agent.predict_ml(models, X.iloc[bs:be]))
        refit_issue = old.issue_no(df.iloc[bs - 1]["回別"])

        for j, t in enumerate(range(bs, be)):
            hist = nums[:t]
            joints = []
            for mi, (cfg, ml) in enumerate(zip(member_cfg, block_ml)):
                probs = {
                    "frequency": frequency_probs(hist, cfg)[None, :, :],
                    "markov": markov_probs(hist, cfg)[None, :, :],
                    "logreg": ml["logreg"][j:j+1],
                    "extra_trees": ml["extra_trees"][j:j+1],
                }
                current = agent.derive_weights(member_losses[mi], cfg["weight_temperature"])
                prev = member_prev_weights[mi]
                model_w = agent.blend_weights(current, prev, cfg["evolution_rate"]) if prev else current
                member_prev_weights[mi] = model_w
                base_pos = agent.ensemble_probs(probs, model_w)[0]
                joints.append(final_joint(base_pos, hist, cfg))
                for name in agent.MODEL_NAMES:
                    loss = digit_nll(nums[t], probs[name][0])
                    member_losses[mi][name] = (1.0 - ewma_alpha) * member_losses[mi][name] + ewma_alpha * loss

            joint = np.tensordot(weights_pool, np.stack(joints), axes=(0, 0))
            joint = np.clip(joint, 1e-15, None)
            joint /= joint.sum()
            actual = old.num3(nums[t])
            ai = int(actual)
            ap = max(float(joint[ai]), 1e-15)
            rank = 1 + int(np.sum(joint > ap))
            top = [n for n, _ in top_numbers(joint, 20)]
            cube = joint.reshape(10, 10, 10)
            marg = np.stack([cube.sum(axis=(1, 2)), cube.sum(axis=(0, 2)), cube.sum(axis=(0, 1))])
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
        "generation": int(champion.get("generation", 1)), "champion_pool_id": pid,
        "champion_pool_members": len(members), "latest_issue": old.issue_no(df.iloc[-1]["回別"]),
        "evaluated_draws": len(result), "warmup": warmup, "refit_interval": refit_interval,
        "audit_draws": audit_draws, "selection": selection, "audit_report_only": audit, "full": full,
        "best_selection_exact_nll": float(best), "improvement_vs_previous_best": float(improvement),
        "improved_best": improved, "stagnation_count": stagnation,
        "replay_version": REPLAY_VERSION, "skipped": False,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    champion["historical_replay"] = {
        "generated_at": now, "dataset_fingerprint": fp, "selection": selection,
        "audit_report_only": audit, "full": full, "stagnation_count": stagnation,
        "replay_version": REPLAY_VERSION, "champion_pool_id": pid,
    }
    champion_path.write_text(json.dumps(champion, ensure_ascii=False, indent=2), encoding="utf-8")

    history_row = {
        "generated_at": now, "champion_id": champion["champion_id"], "generation": champion.get("generation", 1),
        "config_id": pid, "dataset_fingerprint": fp, "latest_issue": replay["latest_issue"],
        "evaluated_draws": len(result), "selection_draws": selection["n"], "audit_draws": audit["n"],
        "selection_exact_nll": selection["exact_nll"], "selection_improvement_vs_uniform": U_EXACT-selection["exact_nll"],
        "selection_top1_rate": selection["top1_rate"], "selection_top20_rate": selection["top20_rate"],
        "selection_mean_rank": selection["mean_rank"], "audit_exact_nll": audit["exact_nll"],
        "audit_top1_rate": audit["top1_rate"], "audit_top20_rate": audit["top20_rate"],
        "audit_mean_rank": audit["mean_rank"], "best_selection_exact_nll": best,
        "improvement_vs_previous_best": improvement, "improved_best": improved, "stagnation_count": stagnation,
    }
    old.append_history(outdir / "replay_history.csv", history_row)
    render_summary(outdir / "latest_accuracy.txt", champion, pool, full, selection, audit, replay, audit_draws, refit_interval)
    print(json.dumps(replay, ensure_ascii=False, indent=2))
    return replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--champion", default="state/evolution_champion.json")
    ap.add_argument("--pool", default="state/champion_pool.json")
    ap.add_argument("--state", default="state/champion_replay_state.json")
    ap.add_argument("--output-dir", default="backtests/champion_replay")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refit-interval", type=int, default=500)
    ap.add_argument("--audit-draws", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_replay(Path(a.csv), Path(a.champion), Path(a.pool), Path(a.state), Path(a.output_dir),
               a.warmup, a.refit_interval, a.audit_draws, force=a.force)


if __name__ == "__main__":
    main()
