#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import numbers3_evo_agent as agent
from numbers3_champion import (
    DEFAULT_CONFIG, FREQUENCY_PROFILES, config_id, fit_ml_models, frequency_probs,
    load_champion, markov_probs, normalize_config,
)

JST = ZoneInfo("Asia/Tokyo")
MODEL_NAMES = agent.MODEL_NAMES
UNIFORM_EXACT_NLL = math.log(1000.0)

SEARCH_SPACE = {
    "half_life_draws": [500, 750, 1000, 1500, 2000, 3000],
    "evolution_rate": [0.40, 0.55, 0.70, 0.85, 1.00],
    "weight_temperature": [0.01, 0.02, 0.04, 0.08],
    "frequency_profile": list(FREQUENCY_PROFILES),
    "markov_blend": [0.45, 0.55, 0.65, 0.75],
    "transition_window": [300, 500, 1000, 2000],
    "lr_c": [0.10, 0.25, 0.50, 1.00],
    "tree_estimators": [120, 180, 240, 320],
    "tree_min_leaf": [4, 8, 12, 20],
}

HISTORY_COLUMNS = [
    "experiment_no", "experiment_id", "generated_at", "dataset_fingerprint",
    "champion_before", "challenger_id", "challenger_config_json",
    "dev_champion_exact_nll", "dev_challenger_exact_nll", "dev_improvement",
    "folds_won", "worst_fold_regression", "confirmation_run", "confirmation_passed",
    "audit_champion_exact_nll", "audit_challenger_exact_nll", "audit_improvement_report_only",
    "decision", "reason", "champion_after",
]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    s = df[["抽せん日", "本数字", "回別"]].fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def exact_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    n = len(y)
    exact_nlls = np.empty(n, dtype=float)
    hit1 = hit20 = 0
    ranks = np.empty(n, dtype=int)
    for i in range(n):
        p = probs[i]
        jp = (p[0, :, None, None] * p[1, None, :, None] * p[2, None, None, :]).reshape(-1)
        actual_idx = int(y[i, 0] * 100 + y[i, 1] * 10 + y[i, 2])
        ap = max(float(jp[actual_idx]), 1e-15)
        exact_nlls[i] = -math.log(ap)
        ranks[i] = 1 + int(np.sum(jp > ap))
        top20 = np.argpartition(jp, -20)[-20:]
        hit20 += int(actual_idx in top20)
        hit1 += int(actual_idx == int(np.argmax(jp)))
    return {
        "n": n,
        "exact_nll": float(exact_nlls.mean()),
        "top1_rate": hit1 / n,
        "top20_rate": hit20 / n,
        "mean_rank": float(ranks.mean()),
    }


def model_probs_for_range(nums, X, start: int, end: int, config: dict, seed: int):
    train_start = 5
    models = fit_ml_models(agent, X.iloc[train_start:start], nums[train_start:start], config, seed=seed)
    ml = agent.predict_ml(models, X.iloc[start:end])
    freq = np.zeros((end-start, 3, 10), dtype=float)
    markov = np.zeros_like(freq)
    for j, t in enumerate(range(start, end)):
        hist = nums[:t]
        freq[j] = frequency_probs(agent, hist, config)
        markov[j] = markov_probs(agent, hist, config)
    return {"frequency": freq, "markov": markov, **ml}


def score_fold_from_probs(nums, probs, range_start: int, fold_start: int, fold_end: int,
                          config: dict, prev_weights=None):
    span = fold_end - fold_start
    cal_n = max(100, span // 2)
    cal_end = min(fold_start + cal_n, fold_end - 50)
    a = fold_start - range_start
    b = fold_end - range_start
    c = cal_end - range_start
    y = nums[fold_start:fold_end]
    local_probs = {name: p[a:b] for name, p in probs.items()}
    cal_len = c - a
    losses = {
        name: agent.mean_position_logloss(y[:cal_len], p[:cal_len])
        for name, p in local_probs.items()
    }
    current = agent.derive_weights(losses, config["weight_temperature"])
    weights = agent.blend_weights(current, prev_weights, config["evolution_rate"])
    ens = agent.ensemble_probs(local_probs, weights)
    metrics = exact_metrics(y[cal_len:], ens[cal_len:])
    metrics["digit_nll"] = agent.mean_position_logloss(y[cal_len:], ens[cal_len:])
    metrics["weights"] = weights
    metrics["fold_start"] = fold_start
    metrics["fold_end"] = fold_end
    return metrics, weights


def evaluate_config(df, nums, X, config: dict, seed: int, audit_draws=1000, fold_size=500, folds=3):
    n = len(df)
    audit_start = n - audit_draws
    dev_start = audit_start - fold_size * folds
    if dev_start < 500:
        raise ValueError("nested evaluation data is insufficient")

    # Strictly past-only: ML is fitted before the development window and then frozen.
    # Heuristic models still update causally before each draw.
    probs = model_probs_for_range(nums, X, dev_start, n, config, seed)

    prev_weights = None
    fold_results = []
    for i in range(folds):
        a = dev_start + i * fold_size
        b = a + fold_size
        m, prev_weights = score_fold_from_probs(nums, probs, dev_start, a, b, config, prev_weights)
        fold_results.append(m)

    dev = {
        "exact_nll": float(np.mean([x["exact_nll"] for x in fold_results])),
        "digit_nll": float(np.mean([x["digit_nll"] for x in fold_results])),
        "top1_rate": float(np.mean([x["top1_rate"] for x in fold_results])),
        "top20_rate": float(np.mean([x["top20_rate"] for x in fold_results])),
        "mean_rank": float(np.mean([x["mean_rank"] for x in fold_results])),
        "fold_exact_nll": [float(x["exact_nll"]) for x in fold_results],
    }

    # Final audit is report-only. Audit outcomes never decide promotion.
    cal_start = audit_start - fold_size
    ca = cal_start - dev_start
    cb = audit_start - dev_start
    aa = audit_start - dev_start
    audit_y = nums[audit_start:n]
    cal_y = nums[cal_start:audit_start]
    losses = {
        name: agent.mean_position_logloss(cal_y, p[ca:cb])
        for name, p in probs.items()
    }
    current = agent.derive_weights(losses, config["weight_temperature"])
    audit_weights = agent.blend_weights(current, prev_weights, config["evolution_rate"])
    audit_probs = {name: p[aa:] for name, p in probs.items()}
    audit_ens = agent.ensemble_probs(audit_probs, audit_weights)
    audit = exact_metrics(audit_y, audit_ens)
    audit["digit_nll"] = agent.mean_position_logloss(audit_y, audit_ens)
    return {"seed": seed, "development": dev, "audit_report_only": audit}


def average_development(seed_results: list[dict]) -> dict:
    return {
        "exact_nll": float(np.mean([r["development"]["exact_nll"] for r in seed_results])),
        "digit_nll": float(np.mean([r["development"]["digit_nll"] for r in seed_results])),
        "fold_exact_nll": list(np.mean(np.array([r["development"]["fold_exact_nll"] for r in seed_results]), axis=0)),
    }


def average_audit(seed_results: list[dict]) -> dict:
    return {
        "exact_nll": float(np.mean([r["audit_report_only"]["exact_nll"] for r in seed_results])),
        "digit_nll": float(np.mean([r["audit_report_only"]["digit_nll"] for r in seed_results])),
        "top1_rate": float(np.mean([r["audit_report_only"]["top1_rate"] for r in seed_results])),
        "top20_rate": float(np.mean([r["audit_report_only"]["top20_rate"] for r in seed_results])),
    }


def compare_dev(champion_results, challenger_results, min_delta=0.0015, max_fold_regression=0.012):
    c = average_development(champion_results)
    x = average_development(challenger_results)
    diffs = np.asarray(c["fold_exact_nll"]) - np.asarray(x["fold_exact_nll"])
    improvement = c["exact_nll"] - x["exact_nll"]
    wins = int(np.sum(diffs > 0))
    worst_regression = max(0.0, float(-np.min(diffs)))
    passed = improvement >= min_delta and wins >= 2 and worst_regression <= max_fold_regression
    return {
        "passed": passed, "improvement": improvement, "folds_won": wins,
        "worst_fold_regression": worst_regression, "champion": c, "challenger": x,
    }


def mutate_config(champion: dict, experiment_no: int, fp: str, seen: set[str]) -> dict:
    base = normalize_config(champion)
    seed_text = f"{fp}:{experiment_no}:{config_id(base)}"
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    keys = list(SEARCH_SPACE)
    for _ in range(100):
        c = dict(base)
        if rng.random() < 0.25:
            count = rng.randint(2, 4)
            for k in rng.sample(keys, count):
                c[k] = rng.choice(SEARCH_SPACE[k])
        else:
            count = 1 if rng.random() < 0.65 else 2
            for k in rng.sample(keys, count):
                vals = SEARCH_SPACE[k]
                cur = c[k]
                if cur in vals:
                    idx = vals.index(cur)
                    choices = vals[max(0, idx-1):min(len(vals), idx+2)]
                    choices = [v for v in choices if v != cur] or vals
                else:
                    choices = vals
                c[k] = rng.choice(choices)
        c = normalize_config(c)
        cid = config_id(c)
        if cid not in seen and cid != config_id(base):
            return c
    c = dict(base)
    k = keys[experiment_no % len(keys)]
    vals = SEARCH_SPACE[k]
    c[k] = vals[(experiment_no // len(keys)) % len(vals)]
    return normalize_config(c)


def read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path)
    for c in HISTORY_COLUMNS:
        if c not in df:
            df[c] = ""
    return df[HISTORY_COLUMNS]


def write_latest(path: Path, row: dict, champion: dict):
    lines = [
        "NUMBERS3 Evolution Engine v2", "="*84,
        f"実験: {row['experiment_id']} / generated={row['generated_at']}",
        f"Champion(before): {row['champion_before']}",
        f"Challenger: {row['challenger_id']}",
        f"判定: {row['decision']}",
        f"理由: {row['reason']}", "",
        "Nested development folds（昇格判定に使用）", "-"*84,
        f"Champion Exact NLL : {row['dev_champion_exact_nll']:.8f}",
        f"Challenger Exact NLL: {row['dev_challenger_exact_nll']:.8f}",
        f"改善量(champion-challenger): {row['dev_improvement']:+.8f}",
        f"fold勝利数: {row['folds_won']}/3 / 最大fold悪化: {row['worst_fold_regression']:.8f}",
        f"確認seed実行: {row['confirmation_run']} / pass={row['confirmation_passed']}", "",
        "Final audit holdout（報告専用・昇格判定には不使用）", "-"*84,
        f"Champion Exact NLL : {row['audit_champion_exact_nll']:.8f}",
        f"Challenger Exact NLL: {row['audit_challenger_exact_nll']:.8f}",
        f"差: {row['audit_improvement_report_only']:+.8f}", "",
        "現在のChampion", "-"*84,
        f"ID: {champion['champion_id']} / generation={champion['generation']}",
        json.dumps(champion['config'], ensure_ascii=False, sort_keys=True), "",
        "注意: Final auditはモデル選択に利用しません。結果を見て昇格させるとholdout汚染になります。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--champion", default="state/evolution_champion.json")
    ap.add_argument("--engine-state", default="state/evolution_engine_state.json")
    ap.add_argument("--history", default="experiments/experiment_history.csv")
    ap.add_argument("--latest", default="experiments/latest_experiment.txt")
    ap.add_argument("--audit-draws", type=int, default=1000)
    ap.add_argument("--fold-size", type=int, default=500)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    champion_path = Path(args.champion)
    state_path = Path(args.engine_state)
    history_path = Path(args.history)
    latest_path = Path(args.latest)
    df, nums = agent.load_history(csv_path)
    fp = dataset_fingerprint(df)
    X = agent.build_features(nums)
    champion = load_champion(champion_path)
    history = read_history(history_path)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    experiment_no = int(state.get("total_experiments", 0)) + 1
    seen = set()
    if not history.empty:
        for raw in history["challenger_config_json"].dropna().astype(str):
            try:
                seen.add(config_id(json.loads(raw)))
            except Exception:
                pass
    challenger_cfg = mutate_config(champion["config"], experiment_no, fp, seen)
    challenger_id = f"chal-{config_id(challenger_cfg)}"
    experiment_id = f"exp-{experiment_no:07d}-{config_id(challenger_cfg)}"

    cached = champion.get("nested_metrics", {})
    if cached.get("dataset_fingerprint") == fp and cached.get("config_id") == config_id(champion["config"]):
        champ_seed42 = cached["seed42"]
        champ_seed137 = cached.get("seed137")
    else:
        champ_seed42 = evaluate_config(df, nums, X, champion["config"], 42, args.audit_draws, args.fold_size)
        champ_seed137 = evaluate_config(df, nums, X, champion["config"], 137, args.audit_draws, args.fold_size)
        champion["nested_metrics"] = {
            "dataset_fingerprint": fp, "config_id": config_id(champion["config"]),
            "seed42": champ_seed42, "seed137": champ_seed137,
        }

    chal_seed42 = evaluate_config(df, nums, X, challenger_cfg, 42, args.audit_draws, args.fold_size)
    first = compare_dev([champ_seed42], [chal_seed42])
    confirmation_run = bool(first["passed"])
    confirmation_passed = False
    chal_seed137 = None
    if confirmation_run:
        chal_seed137 = evaluate_config(df, nums, X, challenger_cfg, 137, args.audit_draws, args.fold_size)
        second = compare_dev([champ_seed42, champ_seed137], [chal_seed42, chal_seed137], min_delta=0.0010)
        confirmation_passed = bool(second["passed"])
        decision_cmp = second
    else:
        decision_cmp = first

    champ_audit = average_audit([champ_seed42] + ([champ_seed137] if champ_seed137 else []))
    chal_audit = average_audit([chal_seed42] + ([chal_seed137] if chal_seed137 else []))
    promoted = confirmation_run and confirmation_passed
    now = datetime.now(JST).isoformat(timespec="seconds")
    champion_before = champion["champion_id"]
    if promoted:
        champion = {
            "champion_id": f"champ-{config_id(challenger_cfg)}",
            "generation": int(champion.get("generation", 1)) + 1,
            "promoted_at": now,
            "source_experiment_id": experiment_id,
            "config": challenger_cfg,
            "nested_metrics": {
                "dataset_fingerprint": fp, "config_id": config_id(challenger_cfg),
                "seed42": chal_seed42, "seed137": chal_seed137,
            },
        }
        reason = "development folds improved on primary and confirmation seeds; final audit was not used for promotion"
        decision = "PROMOTED"
    else:
        reason = (
            "rejected: development improvement/consistency threshold not met"
            if not confirmation_run else
            "rejected: confirmation seed did not reproduce development improvement"
        )
        decision = "REJECTED"

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
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latest(latest_path, row, champion)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
