#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import numbers3_evo_agent as agent
import numbers3_evolution_engine_v3 as v3
from numbers3_champion import load_champion
from numbers3_champion_pool import load_pool, save_pool, seed_from_history, update_pool
from numbers3_v4_models import extended_config, final_joint, v4_config_id

JST = ZoneInfo("Asia/Tokyo")
EVAL_VERSION = "rolling5-bootstrap-metasearch-v4"

# Reuse the proven v3 evaluator, but replace the joint calibrator with the v4 power-calibrated version.
v3.final_joint = final_joint

SEARCH_SPACE = dict(v3.SEARCH_SPACE)
SEARCH_SPACE.update({
    "power_beta": [0.35, 0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25],
})

EXTRA_COLUMNS = [
    "meta_search_strategy", "meta_model_rows", "meta_predicted_nll",
    "meta_uncertainty", "candidate_pool_size", "alpha_spending_threshold",
]
HISTORY_COLUMNS = list(dict.fromkeys(list(v3.HISTORY_COLUMNS) + EXTRA_COLUMNS))


def dataset_fingerprint(df: pd.DataFrame) -> str:
    return v3.dataset_fingerprint(df)


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
    return v3.promotion_experiment_no(champion, history)


def search_mode(since: int) -> tuple[str, int, int]:
    return v3.search_mode(since)


def _random_candidate(base: dict, experiment_no: int, fp: str, since: int, variant: int, seen: set[str]) -> dict:
    cfg0 = extended_config(base)
    mode, min_changes, max_changes = search_mode(since)
    seed = f"v4:{fp}:{experiment_no}:{v4_config_id(cfg0)}:{since}:{variant}"
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    keys = list(SEARCH_SPACE)
    structural_keys = ("power_beta", "uniform_blend", "feature_profile", "autoregressive_mix")
    for _ in range(200):
        c = dict(cfg0)
        count = rng.randint(min_changes, min(max_changes, len(keys)))
        chosen = rng.sample(keys, count)
        if mode in ("global", "structural") and not any(k in chosen for k in structural_keys):
            chosen[0] = rng.choice(structural_keys)
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
        cid = v4_config_id(c)
        if cid != v4_config_id(cfg0) and cid not in seen:
            return c
    c = dict(cfg0)
    k = keys[(experiment_no + variant) % len(keys)]
    vals = list(SEARCH_SPACE[k])
    cur = c.get(k)
    c[k] = vals[(vals.index(cur) + 1) % len(vals)] if cur in vals else vals[0]
    return extended_config(c)


def generate_candidates(base: dict, experiment_no: int, fp: str, since: int, seen: set[str], count: int = 192) -> list[dict]:
    out, local_seen = [], set(seen)
    for variant in range(count * 3):
        c = _random_candidate(base, experiment_no, fp, since, variant, local_seen)
        cid = v4_config_id(c)
        if cid in local_seen:
            continue
        out.append(c)
        local_seen.add(cid)
        if len(out) >= count:
            break
    return out


def _config_frame(configs: list[dict]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        c = extended_config(cfg)
        rows.append({k: c.get(k) for k in SEARCH_SPACE})
    return pd.DataFrame(rows)


def choose_candidate(history: pd.DataFrame, candidates: list[dict], experiment_no: int) -> tuple[dict, dict]:
    if not candidates:
        raise RuntimeError("candidate generation produced no unseen configs")
    records, targets = [], []
    if not history.empty:
        for _, row in history.iterrows():
            try:
                target = float(row["dev_challenger_exact_nll"])
                if not np.isfinite(target):
                    continue
                cfg = extended_config(json.loads(str(row["challenger_config_json"])))
                records.append(cfg)
                targets.append(target)
            except Exception:
                continue

    if len(records) < 200:
        return candidates[0], {
            "strategy": "deterministic_exploration", "rows": len(records),
            "predicted_nll": None, "uncertainty": None, "candidate_pool_size": len(candidates),
        }

    all_cfgs = records + candidates
    frame = _config_frame(all_cfgs)
    categorical = [c for c in frame.columns if frame[c].dtype == object]
    encoded = pd.get_dummies(frame, columns=categorical, dummy_na=True, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train = encoded.iloc[:len(records)].to_numpy(dtype=float)
    X_cand = encoded.iloc[len(records):].to_numpy(dtype=float)
    y = np.asarray(targets, dtype=float)

    age = (len(y) - 1) - np.arange(len(y))
    sw = np.power(0.5, age / 2000.0)
    model = ExtraTreesRegressor(
        n_estimators=240, min_samples_leaf=4, max_features="sqrt",
        random_state=20260823, n_jobs=-1,
    )
    model.fit(X_train, y, sample_weight=sw)
    tree_preds = np.vstack([tree.predict(X_cand) for tree in model.estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)

    if experiment_no % 10 == 0:
        idx = int(np.argmax(std))
        strategy = "meta_uncertainty_exploration"
    else:
        acquisition = mean - 0.35 * std
        idx = int(np.argmin(acquisition))
        strategy = "meta_lcb"
    return candidates[idx], {
        "strategy": strategy, "rows": len(records),
        "predicted_nll": float(mean[idx]), "uncertainty": float(std[idx]),
        "candidate_pool_size": len(candidates),
    }


def evaluate_config(df, nums, cfg, seed, audit_draws=1000, fold_size=300, folds=5):
    return v3.evaluate_config(df, nums, extended_config(cfg), seed, audit_draws, fold_size, folds)


def average_development(results):
    return v3.average_development(results)


def average_audit(results):
    return v3.average_audit(results)


def paired_bootstrap_probability(champion_results, challenger_results, reps: int = 2400) -> float:
    seed_diffs = []
    for c, x in zip(champion_results, challenger_results):
        ca = np.asarray(c["development"]["score_exact_nlls"], dtype=float)
        xa = np.asarray(x["development"]["score_exact_nlls"], dtype=float)
        if len(ca) != len(xa):
            raise ValueError("paired bootstrap requires aligned score rows")
        seed_diffs.append(ca - xa)
    diffs = np.mean(np.vstack(seed_diffs), axis=0)
    rng = np.random.default_rng(20260823 + len(diffs) * 17 + len(champion_results) * 1009)
    n = len(diffs)
    positive = 0
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        positive += float(diffs[idx].mean()) > 0.0
    return positive / reps


def alpha_spending_threshold(since: int, generation: int) -> float:
    if since >= 1000:
        base = 0.9975
    elif since >= 200:
        base = 0.995
    elif since >= 50:
        base = 0.99
    else:
        base = 0.985
    return min(0.999, base + max(0, int(generation) - 3) * 0.0005)


def compare_dev(champion_results, challenger_results, since: int, generation: int,
                min_delta: float = 0.00050) -> dict:
    c = average_development(champion_results)
    x = average_development(challenger_results)
    diffs = np.asarray(c["fold_exact_nll"]) - np.asarray(x["fold_exact_nll"])
    improvement = c["exact_nll"] - x["exact_nll"]
    wins = int(np.sum(diffs > 0))
    worst = max(0.0, float(-np.min(diffs)))
    boot = paired_bootstrap_probability(champion_results, challenger_results)
    threshold = alpha_spending_threshold(since, generation)
    passed = improvement >= min_delta and wins >= 4 and worst <= 0.008 and boot >= threshold
    return {
        "passed": bool(passed), "improvement": float(improvement), "folds_won": wins,
        "worst_fold_regression": worst, "bootstrap_probability": float(boot),
        "bootstrap_threshold": float(threshold), "champion": c, "challenger": x,
    }


def write_latest(path: Path, row: dict, champion: dict, pool: dict) -> None:
    lines = [
        "NUMBERS3 Evolution Engine v4", "=" * 96,
        f"実験: {row['experiment_id']} / generated={row['generated_at']}",
        f"Champion(before): {row['champion_before']}",
        f"Challenger: {row['challenger_id']}",
        f"判定: {row['decision']}",
        f"理由: {row['reason']}",
        f"search={row['search_mode']} / meta={row['meta_search_strategy']} / candidate_pool={row['candidate_pool_size']}",
        f"meta predicted NLL={row['meta_predicted_nll']} / uncertainty={row['meta_uncertainty']}", "",
        "Rolling 5-fold development（昇格判定に使用）", "-" * 96,
        f"Champion Exact NLL : {row['dev_champion_exact_nll']:.8f}",
        f"Challenger Exact NLL: {row['dev_challenger_exact_nll']:.8f}",
        f"改善量: {row['dev_improvement']:+.8f}",
        f"fold勝利数: {row['folds_won']}/{row['fold_count']} / 最大fold悪化: {row['worst_fold_regression']:.8f}",
        f"paired bootstrap P(Challenger better): {row['bootstrap_probability']:.4%}",
        f"必要bootstrap閾値: {row['alpha_spending_threshold']:.4%}",
        f"確認seed実行: {row['confirmation_run']} / pass={row['confirmation_passed']}", "",
        "Final audit holdout（報告専用・昇格判定には不使用）", "-" * 96,
        f"Champion Exact NLL : {row['audit_champion_exact_nll']:.8f}",
        f"Challenger Exact NLL: {row['audit_challenger_exact_nll']:.8f}",
        f"差: {row['audit_improvement_report_only']:+.8f}", "",
        "現在のPrimary Champion", "-" * 96,
        f"ID: {champion['champion_id']} / generation={champion['generation']}",
        json.dumps(champion["config"], ensure_ascii=False, sort_keys=True), "",
        "Champion Pool", "-" * 96,
        f"pool_id={pool.get('pool_id')} / members={len(pool.get('members') or [])}",
    ]
    for m in pool.get("members") or []:
        lines.append(
            f"- {m['champion_id']} cfg={m['config_id']} weight={float(m.get('weight',0)):.4%} "
            f"dev_nll={m.get('dev_exact_nll')}"
        )
    lines += ["", "注意: final auditはモデル選択・meta-search教師には使用しません。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--champion", default="state/evolution_champion.json")
    ap.add_argument("--pool", default="state/champion_pool.json")
    ap.add_argument("--engine-state", default="state/evolution_engine_state.json")
    ap.add_argument("--history", default="experiments/experiment_history.csv")
    ap.add_argument("--latest", default="experiments/latest_experiment.txt")
    ap.add_argument("--meta-latest", default="experiments/meta_search_latest.json")
    ap.add_argument("--audit-draws", type=int, default=1000)
    ap.add_argument("--fold-size", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--candidate-pool-size", type=int, default=192)
    args = ap.parse_args()

    csv_path, champion_path, pool_path = Path(args.csv), Path(args.champion), Path(args.pool)
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
    mode, _, _ = search_mode(since)

    pool = seed_from_history(load_pool(pool_path, champion), history, max_members=5)
    save_pool(pool_path, pool)

    seen: set[str] = set()
    if not history.empty:
        for raw in history["challenger_config_json"].dropna().astype(str):
            try:
                seen.add(v4_config_id(json.loads(raw)))
            except Exception:
                pass
    candidates = generate_candidates(champion["config"], experiment_no, fp, since, seen, args.candidate_pool_size)
    challenger_cfg, meta = choose_candidate(history, candidates, experiment_no)
    challenger_id = f"chal-{v4_config_id(challenger_cfg)}"
    experiment_id = f"exp-{experiment_no:07d}-{v4_config_id(challenger_cfg)}"

    cache = champion.get("nested_metrics_v4", {})
    cache_ok = (
        cache.get("eval_version") == EVAL_VERSION and cache.get("dataset_fingerprint") == fp
        and cache.get("config_id") == v4_config_id(champion["config"])
        and cache.get("fold_size") == args.fold_size and cache.get("folds") == args.folds
    )
    if cache_ok:
        champ42, champ137 = cache["seed42"], cache.get("seed137")
    else:
        champ42 = evaluate_config(df, nums, champion["config"], 42, args.audit_draws, args.fold_size, args.folds)
        champ137 = evaluate_config(df, nums, champion["config"], 137, args.audit_draws, args.fold_size, args.folds)
        champion["nested_metrics_v4"] = {
            "eval_version": EVAL_VERSION, "dataset_fingerprint": fp, "config_id": v4_config_id(champion["config"]),
            "fold_size": args.fold_size, "folds": args.folds, "seed42": champ42, "seed137": champ137,
        }

    chal42 = evaluate_config(df, nums, challenger_cfg, 42, args.audit_draws, args.fold_size, args.folds)
    first = compare_dev([champ42], [chal42], since, champion.get("generation", 1), min_delta=0.00060)
    confirmation_run = bool(first["passed"])
    confirmation_passed = False
    chal137 = None
    decision_cmp = first
    if confirmation_run:
        chal137 = evaluate_config(df, nums, challenger_cfg, 137, args.audit_draws, args.fold_size, args.folds)
        decision_cmp = compare_dev([champ42, champ137], [chal42, chal137], since, champion.get("generation", 1), min_delta=0.00045)
        confirmation_passed = bool(decision_cmp["passed"])

    champ_audit = average_audit([champ42] + ([champ137] if champ137 else []))
    chal_audit = average_audit([chal42] + ([chal137] if chal137 else []))
    promoted = confirmation_run and confirmation_passed
    now = datetime.now(JST).isoformat(timespec="seconds")
    previous_champion = json.loads(json.dumps(champion, ensure_ascii=False))
    champion_before = champion["champion_id"]

    if promoted:
        previous_replay = champion.get("historical_replay")
        champion = {
            "champion_id": f"champ-{v4_config_id(challenger_cfg)}",
            "generation": int(champion.get("generation", 1)) + 1,
            "promoted_at": now,
            "source_experiment_id": experiment_id,
            "config": challenger_cfg,
            "nested_metrics_v4": {
                "eval_version": EVAL_VERSION, "dataset_fingerprint": fp, "config_id": v4_config_id(challenger_cfg),
                "fold_size": args.fold_size, "folds": args.folds, "seed42": chal42, "seed137": chal137,
            },
        }
        if previous_replay:
            champion["previous_champion_replay"] = previous_replay
        pool = update_pool(
            pool, previous_champion, champion,
            decision_cmp["champion"]["exact_nll"], decision_cmp["challenger"]["exact_nll"], max_members=5,
        )
        decision = "PROMOTED"
        reason = "v4 rolling5 + power calibration + confirmation seed + paired bootstrap passed; final audit unused"
    else:
        decision = "REJECTED"
        reason = (
            "rejected: v4 development/consistency/alpha-spending threshold not met"
            if not confirmation_run else
            "rejected: confirmation seed did not reproduce v4 improvement"
        )

    save_pool(pool_path, pool)
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
        "meta_search_strategy": meta["strategy"], "meta_model_rows": meta["rows"],
        "meta_predicted_nll": meta["predicted_nll"], "meta_uncertainty": meta["uncertainty"],
        "candidate_pool_size": meta["candidate_pool_size"],
        "alpha_spending_threshold": decision_cmp["bootstrap_threshold"],
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
        "meta_search_strategy": meta["strategy"], "meta_model_rows": meta["rows"],
        "champion_pool_id": pool.get("pool_id"), "champion_pool_members": len(pool.get("members") or []),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.meta_latest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_latest).write_text(json.dumps({
        "generated_at": now, "experiment_id": experiment_id, "strategy": meta["strategy"],
        "training_rows": meta["rows"], "candidate_pool_size": meta["candidate_pool_size"],
        "selected_config_id": v4_config_id(challenger_cfg), "predicted_nll": meta["predicted_nll"],
        "uncertainty": meta["uncertainty"], "actual_dev_nll": row["dev_challenger_exact_nll"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latest(latest_path, row, champion, pool)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
