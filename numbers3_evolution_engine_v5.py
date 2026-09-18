#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import numbers3_evolution_engine_v4 as v4
from numbers3_v4_models import extended_config

EVAL_VERSION = "rolling7-bootstrap-metasearch-stability-v5"
LAST_STABILITY: dict = {}
_ORIGINAL_WRITE_LATEST = v4.write_latest


def _stability_target(row: pd.Series) -> tuple[dict, float] | None:
    try:
        nll = float(row["dev_challenger_exact_nll"])
        if not np.isfinite(nll):
            return None
        cfg = extended_config(json.loads(str(row["challenger_config_json"])))
        worst = float(pd.to_numeric(row.get("worst_fold_regression", 0.0), errors="coerce"))
        if not np.isfinite(worst):
            worst = 0.0
        folds = float(pd.to_numeric(row.get("fold_count", 5), errors="coerce"))
        wins = float(pd.to_numeric(row.get("folds_won", 0), errors="coerce"))
        if not np.isfinite(folds) or folds <= 0:
            folds = 5.0
        if not np.isfinite(wins):
            wins = 0.0
        miss_fraction = max(0.0, 1.0 - wins / folds)
        target = nll + 0.35 * max(0.0, worst) + 0.0020 * miss_fraction
        return cfg, float(target)
    except Exception:
        return None


def choose_candidate(history: pd.DataFrame, candidates: list[dict], experiment_no: int) -> tuple[dict, dict]:
    if not candidates:
        raise RuntimeError("candidate generation produced no unseen configs")

    records: list[dict] = []
    targets: list[float] = []
    if not history.empty:
        for _, row in history.iterrows():
            parsed = _stability_target(row)
            if parsed is None:
                continue
            cfg, target = parsed
            records.append(cfg)
            targets.append(target)

    if len(records) < 200:
        return candidates[0], {
            "strategy": "stability_deterministic_exploration",
            "rows": len(records),
            "predicted_nll": None,
            "uncertainty": None,
            "candidate_pool_size": len(candidates),
        }

    all_cfgs = records + candidates
    frame = v4._config_frame(all_cfgs)
    categorical = [c for c in frame.columns if frame[c].dtype == object]
    encoded = pd.get_dummies(frame, columns=categorical, dummy_na=True, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train = encoded.iloc[: len(records)].to_numpy(dtype=float)
    X_cand = encoded.iloc[len(records) :].to_numpy(dtype=float)
    y = np.asarray(targets, dtype=float)

    age = (len(y) - 1) - np.arange(len(y))
    sw = np.power(0.5, age / 2200.0)
    model = ExtraTreesRegressor(
        n_estimators=320,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=20260825,
        n_jobs=-1,
    )
    model.fit(X_train, y, sample_weight=sw)
    tree_preds = np.vstack([tree.predict(X_cand) for tree in model.estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)

    if experiment_no % 12 == 0:
        idx = int(np.argmax(std))
        strategy = "stability_meta_uncertainty"
    else:
        acquisition = mean - 0.30 * std
        idx = int(np.argmin(acquisition))
        strategy = "stability_meta_lcb"

    return candidates[idx], {
        "strategy": strategy,
        "rows": len(records),
        "predicted_nll": float(mean[idx]),
        "uncertainty": float(std[idx]),
        "candidate_pool_size": len(candidates),
    }


def compare_dev(champion_results, challenger_results, since: int, generation: int,
                min_delta: float = 0.00050) -> dict:
    global LAST_STABILITY

    c = v4.average_development(champion_results)
    x = v4.average_development(challenger_results)
    diffs = np.asarray(c["fold_exact_nll"], dtype=float) - np.asarray(x["fold_exact_nll"], dtype=float)
    improvement = float(c["exact_nll"] - x["exact_nll"])
    wins = int(np.sum(diffs > 0.0))
    worst_fold = max(0.0, float(-np.min(diffs)))
    boot = float(v4.paired_bootstrap_probability(champion_results, challenger_results))
    threshold = float(v4.alpha_spending_threshold(since, generation))

    groups = np.array_split(diffs, 3)
    regime_improvements = [float(g.mean()) for g in groups]
    regime_wins = int(sum(v > 0.0 for v in regime_improvements))
    recent_improvement = regime_improvements[-1]
    worst_regime = max(0.0, float(-min(regime_improvements)))
    fold_dispersion = float(np.std(diffs))
    required_wins = max(5, len(diffs) - 2)

    passed = (
        improvement >= min_delta
        and wins >= required_wins
        and regime_wins >= 2
        and recent_improvement > 0.0
        and worst_fold <= 0.0060
        and worst_regime <= 0.0030
        and boot >= threshold
    )

    LAST_STABILITY = {
        "folds": int(len(diffs)),
        "required_fold_wins": int(required_wins),
        "regime_improvements": regime_improvements,
        "regime_wins": regime_wins,
        "recent_regime_improvement": recent_improvement,
        "worst_regime_regression": worst_regime,
        "fold_improvement_std": fold_dispersion,
    }
    return {
        "passed": bool(passed),
        "improvement": improvement,
        "folds_won": wins,
        "worst_fold_regression": worst_fold,
        "bootstrap_probability": boot,
        "bootstrap_threshold": threshold,
        "champion": c,
        "challenger": x,
        **LAST_STABILITY,
    }


def _v5_reason(row: dict | pd.Series) -> str:
    decision = str(row.get("decision", ""))
    confirmation_run = str(row.get("confirmation_run", "False")).lower() in ("true", "1")
    confirmation_passed = str(row.get("confirmation_passed", "False")).lower() in ("true", "1")
    if decision == "PROMOTED":
        return "v5 rolling7 + regime stability + power calibration + confirmation seed + paired bootstrap passed; final audit unused"
    if confirmation_run and not confirmation_passed:
        return "rejected: confirmation seed did not reproduce v5 rolling7/regime-stability improvement"
    return "rejected: v5 rolling7/regime-stability/alpha-spending threshold not met"


def write_latest(path: Path, row: dict, champion: dict, pool: dict) -> None:
    row = dict(row)
    is_v5 = str(row.get("eval_version", "")) == EVAL_VERSION
    # v6 reuses this stable report helper. Preserve an explicit newer-version
    # reason instead of overwriting it with a v5 label.
    if is_v5 or not str(row.get("reason", "")).strip():
        row["reason"] = _v5_reason(row)
    _ORIGINAL_WRITE_LATEST(path, row, champion, pool)
    text = path.read_text(encoding="utf-8")
    text = text.replace("NUMBERS3 Evolution Engine v4", "NUMBERS3 Evolution Engine v5 Stability", 1)
    s = LAST_STABILITY or {}
    regime = s.get("regime_improvements") or []
    regime_text = " / ".join(f"{x:+.8f}" for x in regime) if regime else "n/a"
    if is_v5:
        objective_text = "development NLL + worst-fold penalty + fold-loss penalty"
    else:
        objective_text = (
            "development NLL + worst-fold + worst-regime + fold-dispersion "
            "+ fold/regime-loss penalties"
        )
    extra = [
        "",
        "期間安定性ゲート（昇格判定に使用）",
        "-" * 96,
        f"7-fold勝利条件: {row.get('folds_won')}/{row.get('fold_count')} (必要={s.get('required_fold_wins', 'n/a')})",
        f"early / middle / recent 改善量: {regime_text}",
        f"regime勝利数: {s.get('regime_wins', 'n/a')}/3",
        f"recent regime改善量: {s.get('recent_regime_improvement', float('nan')):+.8f}",
        f"worst regime regression: {s.get('worst_regime_regression', float('nan')):.8f}",
        f"fold improvement std: {s.get('fold_improvement_std', float('nan')):.8f}",
        f"Meta-search objective: {objective_text}",
        "目的: 一部期間だけ強い候補ではなく、時系列区間をまたいで安定する候補を優先する。",
    ]
    path.write_text(text.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")


def _cli_value(flag: str, default: str) -> str:
    try:
        i = sys.argv.index(flag)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def fix_history_reason() -> None:
    path = Path(_cli_value("--history", "experiments/experiment_history.csv"))
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "reason" not in df.columns:
        return
    last = df.index[-1]
    if str(df.at[last, "eval_version"]) != EVAL_VERSION:
        return
    df.at[last, "reason"] = _v5_reason(df.loc[last])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    v4.EVAL_VERSION = EVAL_VERSION
    v4.choose_candidate = choose_candidate
    v4.compare_dev = compare_dev
    v4.write_latest = write_latest
    v4.main()
    fix_history_reason()


if __name__ == "__main__":
    main()
