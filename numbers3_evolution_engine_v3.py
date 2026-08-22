#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import numbers3_evo_agent as agent
import numbers3_evolution_engine as v2
from numbers3_champion import load_champion
from numbers3_v3_models import (
    FEATURE_PROFILES,
    build_features,
    extended_config,
    final_joint,
    fit_ml_models,
    frequency_probs,
    markov_probs,
    metrics_from_joint_rows,
    v3_config_id,
)

JST = ZoneInfo("Asia/Tokyo")
EVAL_VERSION = "rolling5-bootstrap-v3"

SEARCH_SPACE = dict(v2.SEARCH_SPACE)
SEARCH_SPACE.update({
    "uniform_blend": [0.25, 0.50, 0.75, 0.90, 1.00],
    "autoregressive_mix": [0.00, 0.05, 0.10, 0.20, 0.35],
    "ar_alpha": [0.5, 1.0, 2.0, 5.0, 10.0],
    "feature_profile": list(FEATURE_PROFILES),
})

HISTORY_COLUMNS = list(v2.HISTORY_COLUMNS) + [
    "eval_version", "fold_count", "bootstrap_probability",
    "experiments_since_last_promotion", "search_mode",
]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    s = df[["抽せん日", "本数字", "回別"]].fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def read_history(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    for c in HISTORY_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[HISTORY_COLUMNS]


def promotion_experiment_no(champion: dict, history: pd.DataFrame) -> int:
    src = str(champion.get("source_experiment_id", ""))
    m = re.search(r"exp-(\d+)", src)
    if m:
        return int(m.group(1))
    if not history.empty:
        x = history[history["decision"].astype(str).eq("PROMOTED")]
        if not x.empty:
            return int(pd.to_numeric(x.iloc[-1]["experiment_no"], errors="coerce"))
    return 0


def search_mode(since: int) -> tuple[str, int, int]:
    if since < 10:
        return "local", 1, 2
    if since < 50:
        return "wider", 2, 3
    if since < 200:
        return "broad", 3, 5
    if since < 1000:
        return "global", 4, 7
    return "structural", 5, 9


def mutate_config(champion_cfg: dict, experiment_no: int, fp: str, seen: set[str], since: int) -> tuple[dict, str]:
    base = extended_config(champion_cfg)
    mode, min_changes, max_changes = search_mode(since)
    rng_seed = f"v3:{fp}:{experiment_no}:{v3_config_id(base)}:{since}"
    rng = random.Random(int(hashlib.sha256(rng_seed.encode()).hexdigest()[:16], 16))
    keys = list(SEARCH_SPACE)

    for _ in range(300):
        c = dict(base)
        count = rng.randint(min_changes, min(max_changes, len(keys)))
        chosen = rng.sample(keys, count)
        if mode == "structural" and not any(k in chosen for k in ("feature_profile", "autoregressive_mix", "uniform_blend")):
            chosen[0] = rng.choice(["feature_profile", "autoregressive_mix", "uniform_blend"])
        for k in chosen:
            vals = list(SEARCH_SPACE[k])
            cur = c.get(k)
            if mode in ("local", "wider") and cur in vals:
                i = vals.index(cur)
                near = [x for x in vals[max(0, i-1):min(len(vals), i+2)] if x != cur]
                choices = near or [x for x in vals if x != cur]
            else:
                choices = [x for x in vals if x != cur]
            if choices:
                c[k] = rng.choice(choices)
        c = extended_config(c)
        cid = v3_config_id(c)
        if cid != v3_config_id(base) and cid not in seen:
            return c, mode

    c = dict(base)
    k = ("uniform_blend", "autoregressive_mix", "feature_profile")[experiment_no % 3]
    vals = list(SEARCH_SPACE[k])
    cur = c[k]
    c[k] = vals[(vals.index(cur) + 1) % len(vals)] if cur in vals else vals[0]
    return extended_config(c), mode


def _component_probs(nums: np.ndarray, X: pd.DataFrame, start: int, end: int, cfg: dict, seed: int) -> dict[str, np.ndarray]:
    train_start = max(agent.LAGS)
    models = fit_ml_models(X.iloc[train_start:start], nums[train_start:start], cfg, seed=seed)
    ml = agent.predict_ml(models, X.iloc[start:end])
    freq = np.zeros((end-start, 3, 10), dtype=float)
    markov = np.zeros_like(freq)
    for j, t in enumerate(range(start, end)):
        hist = nums[:t]
        freq[j] = frequency_probs(hist, cfg)
        markov[j] = markov_probs(hist, cfg)
    return {"frequency": freq, "markov": markov, **ml}


def _score_segment(nums: np.ndarray, probs: dict[str, np.ndarray], start: int, score_start: int,
                   end: int, cfg: dict, prev_weights: dict | None = None) -> tuple[dict, dict]:
    cal_len = score_start - start
    if cal_len < 50:
        raise ValueError("calibration segment is too short")
    y_cal = nums[start:score_start]
    losses = {name: agent.mean_position_logloss(y_cal, p[:cal_len]) for name, p in probs.items()}
    current = agent.derive_weights(losses, cfg["weight_temperature"])
    weights = agent.blend_weights(current, prev_weights, cfg["evolution_rate"]) if prev_weights else current

    joints = []
    for j, t in enumerate(range(score_start, end), start=cal_len):
        one = {name: p[j:j+1] for name, p in probs.items()}
        base_pos = agent.ensemble_probs(one, weights)[0]
        joints.append(final_joint(base_pos, nums[:t], cfg))
    joints_arr = np.asarray(joints, dtype=float)
    m = metrics_from_joint_rows(nums[score_start:end], joints_arr)
    return m, weights


def evaluate_config(df: pd.DataFrame, nums: np.ndarray, cfg_in: dict, seed: int,
                    audit_draws: int = 1000, fold_size: int = 300, folds: int = 5) -> dict:
    cfg = extended_config(cfg_in)
    n = len(df)
    audit_start = n - audit_draws
    dev_start = audit_start - fold_size * folds
    if dev_start < 500:
        raise ValueError("rolling nested evaluation data is insufficient")
    X = build_features(nums, cfg)

    prev_weights = None
    fold_results = []
    for i in range(folds):
        start = dev_start + i * fold_size
        end = start + fold_size
        score_start = start + max(80, fold_size // 2)
        probs = _component_probs(nums, X, start, end, cfg, seed)
        m, prev_weights = _score_segment(nums, probs, start, score_start, end, cfg, prev_weights)
        fold_results.append(m)

    dev_nlls = []
    for x in fold_results:
        dev_nlls.extend(x["score_exact_nlls"])
    dev = {
        "exact_nll": float(np.mean([x["exact_nll"] for x in fold_results])),
        "digit_nll": float(np.mean([x["digit_nll"] for x in fold_results])),
        "top1_rate": float(np.mean([x["top1_rate"] for x in fold_results])),
        "top20_rate": float(np.mean([x["top20_rate"] for x in fold_results])),
        "mean_rank": float(np.mean([x["mean_rank"] for x in fold_results])),
        "fold_exact_nll": [float(x["exact_nll"]) for x in fold_results],
        "score_exact_nlls": dev_nlls,
        "folds": folds,
    }

    cal_start = audit_start - fold_size
    probs = _component_probs(nums, X, cal_start, n, cfg, seed)
    audit_m, _ = _score_segment(nums, probs, cal_start, audit_start, n, cfg, prev_weights)
    audit_m.pop("score_exact_nlls", None)
    return {"seed": seed, "development": dev, "audit_report_only": audit_m, "eval_version": EVAL_VERSION}


def average_development(results: list[dict]) -> dict:
    folds = len(results[0]["development"]["fold_exact_nll"])
    fold_matrix = np.asarray([r["development"]["fold_exact_nll"] for r in results], dtype=float)
    return {
        "exact_nll": float(np.mean([r["development"]["exact_nll"] for r in results])),
        "digit_nll": float(np.mean([r["development"]["digit_nll"] for r in results])),
        "fold_exact_nll": list(fold_matrix.mean(axis=0)),
        "folds": folds,
    }


def average_audit(results: list[dict]) -> dict:
    return {
        "exact_nll": float(np.mean([r["audit_report_only"]["exact_nll"] for r in results])),
        "digit_nll": float(np.mean([r["audit_report_only"]["digit_nll"] for r in results])),
        "top1_rate": float(np.mean([r["audit_report_only"]["top1_rate"] for r in results])),
        "top20_rate": float(np.mean([r["audit_report_only"]["top20_rate"] for r in results])),
    }


def paired_bootstrap_probability(champion_results: list[dict], challenger_results: list[dict], reps: int = 1200) -> float:
    seed_diffs = []
    for c, x in zip(champion_results, challenger_results):
        ca = np.asarray(c["development"]["score_exact_nlls"], dtype=float)
        xa = np.asarray(x["development"]["score_exact_nlls"], dtype=float)
        if len(ca) != len(xa):
            raise ValueError("paired bootstrap requires aligned score rows")
        seed_diffs.append(ca - xa)
    diffs = np.mean(np.vstack(seed_diffs), axis=0)
    rng = np.random.default_rng(20260823 + len(champion_results) * 1009 + len(diffs))
    n = len(diffs)
    positive = 0
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        positive += float(diffs[idx].mean()) > 0.0
    return positive / reps


def compare_dev(champion_results: list[dict], challenger_results: list[dict], since: int,
                min_delta: float, bootstrap_threshold: float) -> dict:
    c = average_development(champion_results)
    x = average_development(challenger_results)
    diffs = np.asarray(c["fold_exact_nll"]) - np.asarray(x["fold_exact_nll"])
    improvement = c["exact_nll"] - x["exact_nll"]
    wins = int(np.sum(diffs > 0))
    worst = max(0.0, float(-np.min(diffs)))
    boot = paired_bootstrap_probability(champion_results, challenger_results)
    required_wins = max(1, c["folds"] - 1)
    adjusted_boot = max(bootstrap_threshold, 0.99 if since >= 1000 else (0.98 if since >= 200 else bootstrap_threshold))
    passed = improvement >= min_delta and wins >= required_wins and worst <= 0.012 and boot >= adjusted_boot
    return {
        "passed": passed, "improvement": float(improvement), "folds_won": wins,
        "worst_fold_regression": worst, "bootstrap_probability": float(boot),
        "bootstrap_threshold": adjusted_boot, "champion": c, "challenger": x,
    }


def write_latest(path: Path, row: dict, champion: dict) -> None:
    lines = [
        "NUMBERS3 Evolution Engine v3", "=" * 92,
        f"実験: {row['experiment_id']} / generated={row['generated_at']}",
        f"Champion(before): {row['champion_before']}",
        f"Challenger: {row['challenger_id']}",
        f"判定: {row['decision']}",
        f"理由: {row['reason']}",
        f"昇格後からの実験数: {row['experiments_since_last_promotion']} / search={row['search_mode']}", "",
        "Rolling 5-fold development（昇格判定に使用）", "-" * 92,
        f"Champion Exact NLL : {row['dev_champion_exact_nll']:.8f}",
        f"Challenger Exact NLL: {row['dev_challenger_exact_nll']:.8f}",
        f"改善量: {row['dev_improvement']:+.8f}",
        f"fold勝利数: {row['folds_won']}/{row['fold_count']} / 最大fold悪化: {row['worst_fold_regression']:.8f}",
        f"paired bootstrap P(Challenger better): {row['bootstrap_probability']:.4%}",
        f"確認seed実行: {row['confirmation_run']} / pass={row['confirmation_passed']}", "",
        "Final audit holdout（報告専用・昇格判定には不使用）", "-" * 92,
        f"Champion Exact NLL : {row['audit_champion_exact_nll']:.8f}",
        f"Challenger Exact NLL: {row['audit_challenger_exact_nll']:.8f}",
        f"差: {row['audit_improvement_report_only']:+.8f}", "",
        "現在のChampion", "-" * 92,
        f"ID: {champion['champion_id']} / generation={champion['generation']}",
        json.dumps(champion["config"], ensure_ascii=False, sort_keys=True), "",
        "注意: final auditはモデル選択に使用しません。adaptive trialの多重比較対策としてbootstrap閾値を強化します。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--champion", default="state/evolution_champion.json")
    ap.add_argument("--engine-state", default="state/evolution_engine_state.json")
    ap.add_argument("--history", default="experiments/experiment_history.csv")
    ap.add_argument("--latest", default="experiments/latest_experiment.txt")
    ap.add_argument("--audit-draws", type=int, default=1000)
    ap.add_argument("--fold-size", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    csv_path, champion_path = Path(args.csv), Path(args.champion)
    state_path, history_path, latest_path = Path(args.engine_state), Path(args.history), Path(args.latest)
    df, nums = agent.load_history(csv_path)
    fp = dataset_fingerprint(df)
    champion = load_champion(champion_path)
    champion["config"] = extended_config(champion.get("config"))
    history = read_history(history_path)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    experiment_no = int(state.get("total_experiments", 0)) + 1
    last_promotion = promotion_experiment_no(champion, history)
    since = max(0, experiment_no - last_promotion - 1)

    seen: set[str] = set()
    if not history.empty:
        for raw in history["challenger_config_json"].dropna().astype(str):
            try:
                seen.add(v3_config_id(json.loads(raw)))
            except Exception:
                pass
    challenger_cfg, mode = mutate_config(champion["config"], experiment_no, fp, seen, since)
    challenger_id = f"chal-{v3_config_id(challenger_cfg)}"
    experiment_id = f"exp-{experiment_no:07d}-{v3_config_id(challenger_cfg)}"

    cache = champion.get("nested_metrics_v3", {})
    cache_ok = (
        cache.get("eval_version") == EVAL_VERSION and cache.get("dataset_fingerprint") == fp
        and cache.get("config_id") == v3_config_id(champion["config"])
        and cache.get("fold_size") == args.fold_size and cache.get("folds") == args.folds
    )
    if cache_ok:
        champ42, champ137 = cache["seed42"], cache.get("seed137")
    else:
        champ42 = evaluate_config(df, nums, champion["config"], 42, args.audit_draws, args.fold_size, args.folds)
        champ137 = evaluate_config(df, nums, champion["config"], 137, args.audit_draws, args.fold_size, args.folds)
        champion["nested_metrics_v3"] = {
            "eval_version": EVAL_VERSION, "dataset_fingerprint": fp, "config_id": v3_config_id(champion["config"]),
            "fold_size": args.fold_size, "folds": args.folds, "seed42": champ42, "seed137": champ137,
        }

    chal42 = evaluate_config(df, nums, challenger_cfg, 42, args.audit_draws, args.fold_size, args.folds)
    first = compare_dev([champ42], [chal42], since, min_delta=0.0010, bootstrap_threshold=0.97)
    confirmation_run = bool(first["passed"])
    confirmation_passed = False
    chal137 = None
    decision_cmp = first
    if confirmation_run:
        chal137 = evaluate_config(df, nums, challenger_cfg, 137, args.audit_draws, args.fold_size, args.folds)
        decision_cmp = compare_dev([champ42, champ137], [chal42, chal137], since,
                                   min_delta=0.00075, bootstrap_threshold=0.975)
        confirmation_passed = bool(decision_cmp["passed"])

    champ_audit = average_audit([champ42] + ([champ137] if champ137 else []))
    chal_audit = average_audit([chal42] + ([chal137] if chal137 else []))
    promoted = confirmation_run and confirmation_passed
    now = datetime.now(JST).isoformat(timespec="seconds")
    champion_before = champion["champion_id"]

    if promoted:
        previous_replay = champion.get("historical_replay")
        champion = {
            "champion_id": f"champ-{v3_config_id(challenger_cfg)}",
            "generation": int(champion.get("generation", 1)) + 1,
            "promoted_at": now,
            "source_experiment_id": experiment_id,
            "config": challenger_cfg,
            "nested_metrics_v3": {
                "eval_version": EVAL_VERSION, "dataset_fingerprint": fp, "config_id": v3_config_id(challenger_cfg),
                "fold_size": args.fold_size, "folds": args.folds, "seed42": chal42, "seed137": chal137,
            },
        }
        if previous_replay:
            champion["previous_champion_replay"] = previous_replay
        decision = "PROMOTED"
        reason = "rolling 5-fold + confirmation seed + paired bootstrap passed; final audit not used"
    else:
        decision = "REJECTED"
        if not confirmation_run:
            reason = "rejected: rolling-fold improvement/consistency/bootstrap threshold not met"
        else:
            reason = "rejected: confirmation seed/bootstrap did not reproduce improvement"

    row = {
        "experiment_no": experiment_no, "experiment_id": experiment_id, "generated_at": now,
        "dataset_fingerprint": fp, "champion_before": champion_before, "challenger_id": challenger_id,
        "challenger_config_json": json.dumps(challenger_cfg, ensure_ascii=False, sort_keys=True),
        "dev_champion_exact_nll": decision_cmp["champion"]["exact_nll"],
        "dev_challenger_exact_nll": decision_cmp["challenger"]["exact_nll"],
        "dev_improvement": decision_cmp["improvement"], "folds_won": decision_cmp["folds_won"],
        "worst_fold_regression": decision_cmp["worst_fold_regression"],
        "confirmation_run": confirmation_run, "confirmation_passed": confirmation_passed,
        "audit_champion_exact_nll": champ_audit["exact_nll"],
        "audit_challenger_exact_nll": chal_audit["exact_nll"],
        "audit_improvement_report_only": champ_audit["exact_nll"] - chal_audit["exact_nll"],
        "decision": decision, "reason": reason, "champion_after": champion["champion_id"],
        "eval_version": EVAL_VERSION, "fold_count": args.folds,
        "bootstrap_probability": decision_cmp["bootstrap_probability"],
        "experiments_since_last_promotion": since, "search_mode": mode,
    }

    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.tail(20000).to_csv(history_path, index=False, encoding="utf-8-sig")
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    champion_path.write_text(json.dumps(champion, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {
        "total_experiments": experiment_no, "last_experiment_id": experiment_id,
        "last_completed_at": now, "last_dataset_fingerprint": fp,
        "current_champion_id": champion["champion_id"], "current_generation": champion["generation"],
        "experiments_since_last_promotion": 0 if promoted else since + 1,
        "search_mode": "local" if promoted else mode, "eval_version": EVAL_VERSION,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latest(latest_path, row, champion)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
