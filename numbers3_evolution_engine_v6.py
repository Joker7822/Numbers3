#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import numbers3_evolution_engine_v4 as v4
import numbers3_evolution_engine_v5 as v5
from numbers3_champion import load_champion
from numbers3_v4_models import extended_config, v4_config_id

JST = ZoneInfo("Asia/Tokyo")
EVAL_VERSION = "rolling7-bootstrap-metasearch-structural-diversity-v6"
SHADOW_POOL_VERSION = "champion-pool-shadow-v1"
LAST_COMPARISON: dict[str, Any] = {}

SEARCH_SPACE = dict(v4.SEARCH_SPACE)
SEARCH_SPACE.update({
    "interaction_mix": [0.00, 0.05, 0.10, 0.20, 0.30],
    "interaction_alpha": [0.5, 1.0, 2.0, 5.0, 10.0],
    "regime_mix": [0.00, 0.05, 0.10, 0.20, 0.30],
    "regime_window": [100, 300, 700, 1500],
})

EXTRA_HISTORY_COLUMNS = [
    "regime_early_improvement",
    "regime_middle_improvement",
    "regime_recent_improvement",
    "regime_wins",
    "worst_regime_regression",
    "fold_improvement_std",
    "model_family",
    "shadow_pool_eligible",
]
HISTORY_COLUMNS = list(dict.fromkeys(list(v4.HISTORY_COLUMNS) + EXTRA_HISTORY_COLUMNS))
STRUCTURAL_KEYS = (
    "feature_profile",
    "uniform_blend",
    "autoregressive_mix",
    "power_beta",
    "interaction_mix",
    "regime_mix",
    "regime_window",
    "half_life_draws",
    "markov_blend",
    "transition_window",
)
NEW_FAMILY_KEYS = ("interaction_mix", "regime_mix")


def model_family(config: dict[str, Any]) -> str:
    cfg = extended_config(config)
    pair = float(cfg.get("interaction_mix", 0.0)) > 0.0
    regime = float(cfg.get("regime_mix", 0.0)) > 0.0
    if pair and regime:
        return "pairwise_regime_hybrid"
    if pair:
        return "pairwise_interaction"
    if regime:
        return "recent_regime_pairwise"
    return "legacy_v5_family"


def _random_candidate(
    base: dict,
    experiment_no: int,
    fp: str,
    since: int,
    variant: int,
    seen: set[str],
) -> dict:
    cfg0 = extended_config(base)
    mode, min_changes, max_changes = v4.search_mode(since)
    seed = f"v6:{fp}:{experiment_no}:{v4_config_id(cfg0)}:{since}:{variant}"
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    keys = list(SEARCH_SPACE)
    structural_keys = (
        "power_beta", "uniform_blend", "feature_profile", "autoregressive_mix",
        "interaction_mix", "regime_mix", "regime_window",
    )

    for _ in range(240):
        candidate = dict(cfg0)
        count = rng.randint(min_changes, min(max_changes, len(keys)))
        chosen = rng.sample(keys, count)

        if mode in ("global", "structural") and not any(k in chosen for k in structural_keys):
            chosen[0] = rng.choice(structural_keys)

        # The current Champion has been unchanged for a very long plateau. In
        # structural mode, two thirds of generated candidates are forced to
        # exercise at least one genuinely new family rather than merely adding
        # more trials around the legacy v5 neighbourhood.
        force_new_family = since >= 1000 and variant % 3 != 2
        if force_new_family and not any(k in chosen for k in NEW_FAMILY_KEYS):
            chosen[0] = rng.choice(NEW_FAMILY_KEYS)

        for key in chosen:
            values = list(SEARCH_SPACE[key])
            current = candidate.get(key)
            if mode in ("local", "wider") and current in values:
                i = values.index(current)
                nearby = [
                    x for x in values[max(0, i - 1):min(len(values), i + 2)]
                    if x != current
                ]
                choices = nearby or [x for x in values if x != current]
            else:
                choices = [x for x in values if x != current]

            if force_new_family and key in NEW_FAMILY_KEYS:
                nonzero = [x for x in choices if float(x) > 0.0]
                if nonzero:
                    choices = nonzero
            if choices:
                candidate[key] = rng.choice(choices)

        candidate = extended_config(candidate)
        cid = v4_config_id(candidate)
        if cid != v4_config_id(cfg0) and cid not in seen:
            return candidate

    # Deterministic escape hatch always enters a new structural family.
    candidate = dict(cfg0)
    key = NEW_FAMILY_KEYS[(experiment_no + variant) % len(NEW_FAMILY_KEYS)]
    candidate[key] = 0.10
    return extended_config(candidate)


def generate_candidates(
    base: dict,
    experiment_no: int,
    fp: str,
    since: int,
    seen: set[str],
    count: int = 192,
) -> list[dict]:
    out: list[dict] = []
    local_seen = set(seen)
    for variant in range(count * 4):
        cfg = _random_candidate(base, experiment_no, fp, since, variant, local_seen)
        cid = v4_config_id(cfg)
        if cid in local_seen:
            continue
        out.append(cfg)
        local_seen.add(cid)
        if len(out) >= count:
            break
    return out


def _stability_target(row: pd.Series) -> tuple[dict, float] | None:
    try:
        nll = float(row["dev_challenger_exact_nll"])
        if not np.isfinite(nll):
            return None
        cfg = extended_config(json.loads(str(row["challenger_config_json"])))

        worst_fold = float(pd.to_numeric(row.get("worst_fold_regression", 0.0), errors="coerce"))
        if not np.isfinite(worst_fold):
            worst_fold = 0.0
        folds = float(pd.to_numeric(row.get("fold_count", 7), errors="coerce"))
        wins = float(pd.to_numeric(row.get("folds_won", 0), errors="coerce"))
        if not np.isfinite(folds) or folds <= 0:
            folds = 7.0
        if not np.isfinite(wins):
            wins = 0.0
        miss_fraction = max(0.0, 1.0 - wins / folds)

        worst_regime = float(pd.to_numeric(row.get("worst_regime_regression", 0.0), errors="coerce"))
        fold_std = float(pd.to_numeric(row.get("fold_improvement_std", 0.0), errors="coerce"))
        regime_wins = float(pd.to_numeric(row.get("regime_wins", 3), errors="coerce"))
        worst_regime = 0.0 if not np.isfinite(worst_regime) else max(0.0, worst_regime)
        fold_std = 0.0 if not np.isfinite(fold_std) else max(0.0, fold_std)
        regime_wins = 3.0 if not np.isfinite(regime_wins) else regime_wins
        regime_miss_fraction = max(0.0, 1.0 - regime_wins / 3.0)

        target = (
            nll
            + 0.40 * max(0.0, worst_fold)
            + 0.30 * worst_regime
            + 0.15 * fold_std
            + 0.0020 * miss_fraction
            + 0.0010 * regime_miss_fraction
        )
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
        selected = candidates[0]
        return selected, {
            "strategy": "v6_structural_deterministic_exploration",
            "rows": len(records),
            "predicted_nll": None,
            "uncertainty": None,
            "candidate_pool_size": len(candidates),
            "selected_family": model_family(selected),
        }

    all_cfgs = records + candidates
    frame = v4._config_frame(all_cfgs)
    categorical = [c for c in frame.columns if frame[c].dtype == object]
    encoded = pd.get_dummies(frame, columns=categorical, dummy_na=True, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train = encoded.iloc[: len(records)].to_numpy(dtype=float)
    X_cand = encoded.iloc[len(records):].to_numpy(dtype=float)
    y = np.asarray(targets, dtype=float)

    age = (len(y) - 1) - np.arange(len(y))
    sample_weight = np.power(0.5, age / 2500.0)
    model = ExtraTreesRegressor(
        n_estimators=360,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=20260903,
        n_jobs=-1,
    )
    model.fit(X_train, y, sample_weight=sample_weight)
    tree_preds = np.vstack([tree.predict(X_cand) for tree in model.estimators_])
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)

    families = np.array([model_family(c) for c in candidates], dtype=object)
    is_new = families != "legacy_v5_family"

    if experiment_no % 10 == 0 and np.any(is_new):
        eligible = np.where(is_new)[0]
        idx = int(eligible[np.argmax(std[eligible])])
        strategy = "v6_new_family_uncertainty"
    elif experiment_no % 12 == 0:
        idx = int(np.argmax(std))
        strategy = "v6_meta_uncertainty"
    else:
        # Small novelty credit prevents the entrenched legacy family from
        # monopolising the LCB while still requiring predicted stability.
        novelty = np.where(is_new, 0.00015, 0.0)
        acquisition = mean - 0.30 * std - novelty
        idx = int(np.argmin(acquisition))
        strategy = "v6_stability_meta_lcb"

    return candidates[idx], {
        "strategy": strategy,
        "rows": len(records),
        "predicted_nll": float(mean[idx]),
        "uncertainty": float(std[idx]),
        "candidate_pool_size": len(candidates),
        "selected_family": str(families[idx]),
    }


def compare_dev(champion_results, challenger_results, since: int, generation: int,
                min_delta: float = 0.00050) -> dict:
    global LAST_COMPARISON
    result = v5.compare_dev(champion_results, challenger_results, since, generation, min_delta=min_delta)
    LAST_COMPARISON = dict(result)
    return result


def _v6_reason(row: dict | pd.Series) -> str:
    decision = str(row.get("decision", ""))
    confirmation_run = str(row.get("confirmation_run", "False")).lower() in ("true", "1")
    confirmation_passed = str(row.get("confirmation_passed", "False")).lower() in ("true", "1")
    if decision == "PROMOTED":
        return "v6 rolling7 + regime stability + structural-family search + confirmation seed + paired bootstrap passed; final audit unused"
    if confirmation_run and not confirmation_passed:
        return "rejected: confirmation seed did not reproduce v6 rolling7/regime-stability improvement"
    return "rejected: v6 rolling7/regime-stability/alpha-spending threshold not met"


def write_latest(path: Path, row: dict, champion: dict, pool: dict) -> None:
    row = dict(row)
    row["reason"] = _v6_reason(row)
    v5.write_latest(path, row, champion, pool)
    text = path.read_text(encoding="utf-8")
    text = text.replace("NUMBERS3 Evolution Engine v5 Stability", "NUMBERS3 Evolution Engine v6 Structural Diversity", 1)
    text = text.replace("Rolling 5-fold development", "Rolling 7-fold development", 1)
    try:
        cfg = json.loads(str(row["challenger_config_json"]))
        family = model_family(cfg)
    except Exception:
        family = "unknown"
    extra = [
        "",
        "v6 structural family",
        "-" * 96,
        f"challenger family: {family}",
        "pairwise_interaction: 3つの桁ペア分布から低分散jointを構成（causal）",
        "recent_regime_pairwise: 直近windowの桁ペア分布を縮小推定して混合（causal）",
        "Primary昇格条件: v5と同じ厳格な7-fold/regime/bootstrap条件を維持",
        "Rejected候補: active Champion Poolには直接入れずshadow候補としてのみ監査",
    ]
    path.write_text(text.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")


def _cli_value(flag: str, default: str) -> str:
    try:
        idx = sys.argv.index(flag)
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return default


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _structural_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ca = extended_config(a)
    cb = extended_config(b)
    changed = 0
    for key in STRUCTURAL_KEYS:
        av = ca.get(key, 0.0 if key.endswith("_mix") else None)
        bv = cb.get(key, 0.0 if key.endswith("_mix") else None)
        try:
            same = abs(float(av) - float(bv)) <= 1e-12
        except (TypeError, ValueError):
            same = av == bv
        changed += int(not same)
    return changed / max(1, len(STRUCTURAL_KEYS))


def _fix_history_last_row() -> pd.DataFrame:
    path = Path(_cli_value("--history", "experiments/experiment_history.csv"))
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if df.empty:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df

    last = df.index[-1]
    if str(df.at[last, "eval_version"]) == EVAL_VERSION:
        stability = v5.LAST_STABILITY or LAST_COMPARISON or {}
        regime = list(stability.get("regime_improvements") or [])
        while len(regime) < 3:
            regime.append(float("nan"))
        df.at[last, "reason"] = _v6_reason(df.loc[last])
        df.at[last, "regime_early_improvement"] = regime[0]
        df.at[last, "regime_middle_improvement"] = regime[1]
        df.at[last, "regime_recent_improvement"] = regime[2]
        df.at[last, "regime_wins"] = stability.get("regime_wins", "")
        df.at[last, "worst_regime_regression"] = stability.get("worst_regime_regression", "")
        df.at[last, "fold_improvement_std"] = stability.get("fold_improvement_std", "")
        try:
            cfg = json.loads(str(df.at[last, "challenger_config_json"]))
            df.at[last, "model_family"] = model_family(cfg)
        except Exception:
            df.at[last, "model_family"] = "unknown"

        improvement = float(pd.to_numeric(df.at[last, "dev_improvement"], errors="coerce"))
        folds_won = float(pd.to_numeric(df.at[last, "folds_won"], errors="coerce"))
        worst_fold = float(pd.to_numeric(df.at[last, "worst_fold_regression"], errors="coerce"))
        bootstrap = float(pd.to_numeric(df.at[last, "bootstrap_probability"], errors="coerce"))
        regime_wins = float(pd.to_numeric(df.at[last, "regime_wins"], errors="coerce"))
        worst_regime = float(pd.to_numeric(df.at[last, "worst_regime_regression"], errors="coerce"))
        eligible = (
            np.isfinite(improvement) and improvement >= 0.00050
            and np.isfinite(folds_won) and folds_won >= 5
            and np.isfinite(worst_fold) and worst_fold <= 0.0060
            and np.isfinite(bootstrap) and bootstrap >= 0.90
            and np.isfinite(regime_wins) and regime_wins >= 2
            and np.isfinite(worst_regime) and worst_regime <= 0.0030
        )
        df.at[last, "shadow_pool_eligible"] = bool(eligible)

    df = df[HISTORY_COLUMNS]
    df.tail(20000).to_csv(path, index=False, encoding="utf-8-sig")
    return df.tail(20000).copy()


def _shadow_score(row: pd.Series) -> float:
    nll = float(pd.to_numeric(row.get("dev_challenger_exact_nll"), errors="coerce"))
    worst_fold = float(pd.to_numeric(row.get("worst_fold_regression"), errors="coerce"))
    worst_regime = float(pd.to_numeric(row.get("worst_regime_regression"), errors="coerce"))
    fold_std = float(pd.to_numeric(row.get("fold_improvement_std"), errors="coerce"))
    values = [nll, worst_fold, worst_regime, fold_std]
    if not all(np.isfinite(x) for x in values):
        return float("inf")
    return nll + 0.35 * max(0.0, worst_fold) + 0.25 * max(0.0, worst_regime) + 0.10 * max(0.0, fold_std)


def _update_shadow_pool(history: pd.DataFrame) -> int:
    out_path = Path("state/champion_pool_candidates.json")
    champion_path = Path(_cli_value("--champion", "state/evolution_champion.json"))
    champion = load_champion(champion_path)
    champion_cfg = extended_config(champion.get("config"))

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not history.empty:
        recent = history.tail(6000).copy()
        eligible_rows = recent[recent["shadow_pool_eligible"].map(_bool_value)]
        scored: list[tuple[float, pd.Series]] = []
        for _, row in eligible_rows.iterrows():
            score = _shadow_score(row)
            if np.isfinite(score):
                scored.append((score, row))
        scored.sort(key=lambda x: x[0])

        for score, row in scored:
            try:
                cfg = extended_config(json.loads(str(row["challenger_config_json"])))
                cid = v4_config_id(cfg)
            except Exception:
                continue
            if cid in seen:
                continue
            distance_primary = _structural_distance(champion_cfg, cfg)
            if candidates:
                distance_selected = min(_structural_distance(cfg, c["config"]) for c in candidates)
            else:
                distance_selected = 1.0
            # Prefer meaningful structural diversity; if very few candidates are
            # available, a strong near-neighbour may still be retained in shadow.
            if distance_primary < 0.20 and distance_selected < 0.20 and len(candidates) >= 3:
                continue
            seen.add(cid)
            candidates.append({
                "candidate_id": f"shadow-{cid}",
                "config_id": cid,
                "experiment_id": str(row.get("experiment_id", "")),
                "generated_at": str(row.get("generated_at", "")),
                "model_family": model_family(cfg),
                "config": cfg,
                "dev_exact_nll": float(pd.to_numeric(row.get("dev_challenger_exact_nll"), errors="coerce")),
                "dev_improvement": float(pd.to_numeric(row.get("dev_improvement"), errors="coerce")),
                "folds_won": int(float(pd.to_numeric(row.get("folds_won"), errors="coerce"))),
                "bootstrap_probability": float(pd.to_numeric(row.get("bootstrap_probability"), errors="coerce")),
                "worst_fold_regression": float(pd.to_numeric(row.get("worst_fold_regression"), errors="coerce")),
                "regime_wins": int(float(pd.to_numeric(row.get("regime_wins"), errors="coerce"))),
                "worst_regime_regression": float(pd.to_numeric(row.get("worst_regime_regression"), errors="coerce")),
                "structural_distance_from_primary": distance_primary,
                "shadow_score": float(score),
                "active_for_prediction": False,
            })
            if len(candidates) >= 8:
                break

    payload = {
        "version": SHADOW_POOL_VERSION,
        "updated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "primary_champion_id": champion.get("champion_id"),
        "policy": "SHADOW_ONLY: rejected candidates never affect live prediction; admission requires stable dev evidence and structural diversity",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(candidates)


def _update_engine_state(shadow_count: int) -> None:
    path = Path(_cli_value("--engine-state", "state/evolution_engine_state.json"))
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    state["structural_model_version"] = "pairwise-regime-v6"
    state["shadow_pool_candidates"] = int(shadow_count)
    state["active_pool_policy"] = "promoted-only-diversity-aware"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # v4.main remains the proven orchestration path. v6 only replaces candidate
    # generation, the stability-aware meta objective, reporting, and expands the
    # causal final_joint family through numbers3_v4_models.
    v4.EVAL_VERSION = EVAL_VERSION
    v4.SEARCH_SPACE = SEARCH_SPACE
    v4.HISTORY_COLUMNS = HISTORY_COLUMNS
    v4.generate_candidates = generate_candidates
    v4.choose_candidate = choose_candidate
    v4.compare_dev = compare_dev
    v4.write_latest = write_latest

    # evaluate_config in the reused v3 evaluator resolves final_joint through
    # this module-global hook set by v4, so the optional v6 components are used
    # in every walk-forward fold without changing causal boundaries.
    v4.main()
    history = _fix_history_last_row()
    shadow_count = _update_shadow_pool(history)
    _update_engine_state(shadow_count)


if __name__ == "__main__":
    main()
