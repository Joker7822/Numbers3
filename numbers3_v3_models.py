#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import numbers3_evo_agent as agent
from numbers3_champion import (
    config_id,
    fit_ml_models as champion_fit_ml_models,
    frequency_probs as champion_frequency_probs,
    markov_probs as champion_markov_probs,
    normalize_config as base_normalize_config,
)

EPS = 1e-15
U_EXACT = math.log(1000.0)
V3_DEFAULTS = {
    "uniform_blend": 1.0,
    "autoregressive_mix": 0.0,
    "ar_alpha": 2.0,
    "feature_profile": "base",
}
FEATURE_PROFILES = ("base", "compact", "gap", "gap_pair", "gap_pair_shape")


def extended_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = base_normalize_config(config)
    for k, v in V3_DEFAULTS.items():
        cfg.setdefault(k, v)
    cfg["uniform_blend"] = float(cfg["uniform_blend"])
    cfg["autoregressive_mix"] = float(cfg["autoregressive_mix"])
    cfg["ar_alpha"] = float(cfg["ar_alpha"])
    cfg["feature_profile"] = str(cfg["feature_profile"])
    if not (0.0 <= cfg["uniform_blend"] <= 1.0):
        raise ValueError("uniform_blend must be in [0,1]")
    if not (0.0 <= cfg["autoregressive_mix"] <= 1.0):
        raise ValueError("autoregressive_mix must be in [0,1]")
    if cfg["ar_alpha"] <= 0:
        raise ValueError("ar_alpha must be > 0")
    if cfg["feature_profile"] not in FEATURE_PROFILES:
        raise ValueError(f"unknown feature_profile={cfg['feature_profile']}")
    return cfg


def v3_config_id(config: dict[str, Any]) -> str:
    return config_id(extended_config(config))


def _gap_features(nums: np.ndarray) -> pd.DataFrame:
    n = len(nums)
    last = np.full((3, 10), -1, dtype=int)
    values: dict[str, np.ndarray] = {}
    for pos in range(3):
        for d in range(10):
            values[f"gap_p{pos}_d{d}"] = np.empty(n, dtype=float)
    for t in range(n):
        for pos in range(3):
            for d in range(10):
                age = 1000 if last[pos, d] < 0 else min(1000, t - last[pos, d])
                values[f"gap_p{pos}_d{d}"][t] = age / 1000.0
        for pos in range(3):
            d = int(nums[t, pos])
            if 0 <= d <= 9:
                last[pos, d] = t
    return pd.DataFrame(values)


def _pair_features(nums: np.ndarray) -> pd.DataFrame:
    n = len(nums)
    idx = pd.RangeIndex(n)
    out: dict[str, pd.Series] = {}
    valid01 = (nums[:, 0] >= 0) & (nums[:, 1] >= 0)
    valid12 = (nums[:, 1] >= 0) & (nums[:, 2] >= 0)
    p01 = np.where(valid01, nums[:, 0] * 10 + nums[:, 1], -1)
    p12 = np.where(valid12, nums[:, 1] * 10 + nums[:, 2], -1)
    s01 = pd.Series(p01, index=idx).shift(1)
    s12 = pd.Series(p12, index=idx).shift(1)
    for code in range(100):
        out[f"prev_pair01_{code:02d}"] = (s01 == code).astype(float)
        out[f"prev_pair12_{code:02d}"] = (s12 == code).astype(float)
    return pd.DataFrame(out, index=idx)


def _rolling_shape_features(nums: np.ndarray) -> pd.DataFrame:
    n = len(nums)
    idx = pd.RangeIndex(n)
    valid = (nums >= 0).all(axis=1)
    safe = np.where(nums >= 0, nums, 0)
    raw = {
        "sum": safe.sum(axis=1) / 27.0,
        "unique": np.array([len(set(r)) if ok else 0 for r, ok in zip(safe, valid)], dtype=float) / 3.0,
        "odd": (safe % 2).sum(axis=1) / 3.0,
        "spread": (safe.max(axis=1) - safe.min(axis=1)) / 9.0,
        "repeat": ((safe[:, 0] == safe[:, 1]) | (safe[:, 1] == safe[:, 2]) | (safe[:, 0] == safe[:, 2])).astype(float),
    }
    out: dict[str, pd.Series] = {}
    for name, arr in raw.items():
        s = pd.Series(arr, index=idx)
        for w in (30, 100, 300):
            out[f"shape_{name}_mean_w{w}"] = s.shift(1).rolling(w, min_periods=5).mean()
    return pd.DataFrame(out, index=idx)


def build_features(nums: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    cfg = extended_config(config)
    base = agent.build_features(nums).reset_index(drop=True)
    profile = cfg["feature_profile"]
    if profile == "compact":
        keep = []
        for c in base.columns:
            if "_lag1_" in c or "_lag2_" in c or "_lag3_" in c:
                keep.append(c)
            elif "_freq_w100_" in c or "_freq_w300_" in c:
                keep.append(c)
            elif c.startswith("prev_"):
                keep.append(c)
        base = base[keep]
    if profile in ("gap", "gap_pair", "gap_pair_shape"):
        base = pd.concat([base, _gap_features(nums)], axis=1)
    if profile in ("gap_pair", "gap_pair_shape"):
        base = pd.concat([base, _pair_features(nums)], axis=1)
    if profile == "gap_pair_shape":
        base = pd.concat([base, _rolling_shape_features(nums)], axis=1)
    return base.fillna(0.1)


def make_next_features(nums: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    placeholder = np.array([[-1, -1, -1]], dtype=int)
    return build_features(np.vstack([nums, placeholder]), config).iloc[[-1]].copy()


def fit_ml_models(X_train: pd.DataFrame, y_train: np.ndarray, config: dict[str, Any], seed: int | None = None):
    return champion_fit_ml_models(agent, X_train, y_train, extended_config(config), seed=seed)


def frequency_probs(history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return champion_frequency_probs(agent, history, extended_config(config))


def markov_probs(history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return champion_markov_probs(agent, history, extended_config(config))


def position_joint(prob3: np.ndarray) -> np.ndarray:
    p = agent.normalize_probs(np.asarray(prob3, dtype=float))
    joint = (p[0, :, None, None] * p[1, None, :, None] * p[2, None, None, :]).reshape(-1)
    joint = np.clip(joint, EPS, None)
    return joint / joint.sum()


def joint_marginals(joint: np.ndarray) -> np.ndarray:
    j = np.asarray(joint, dtype=float).reshape(10, 10, 10)
    out = np.stack([j.sum(axis=(1, 2)), j.sum(axis=(0, 2)), j.sum(axis=(0, 1))])
    return agent.normalize_probs(out)


def autoregressive_joint(history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = extended_config(config)
    if len(history) == 0:
        return np.full(1000, 0.001, dtype=float)
    h = np.asarray(history, dtype=int)
    h = h[(h >= 0).all(axis=1) & (h <= 9).all(axis=1)]
    if len(h) == 0:
        return np.full(1000, 0.001, dtype=float)
    weights = agent.recency_sample_weight(len(h), cfg["half_life_draws"])
    alpha = cfg["ar_alpha"]
    c1 = np.full(10, alpha, dtype=float)
    c2 = np.full((10, 10), alpha, dtype=float)
    c3 = np.full((10, 10, 10), alpha, dtype=float)
    for row, w in zip(h, weights):
        a, b, c = map(int, row)
        c1[a] += w
        c2[a, b] += w
        c3[a, b, c] += w
    p1 = c1 / c1.sum()
    p2 = c2 / c2.sum(axis=1, keepdims=True)
    p3 = c3 / c3.sum(axis=2, keepdims=True)
    joint = (p1[:, None, None] * p2[:, :, None] * p3).reshape(-1)
    joint = np.clip(joint, EPS, None)
    return joint / joint.sum()


def final_joint(base_position_probs: np.ndarray, history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = extended_config(config)
    base = position_joint(base_position_probs)
    ar_mix = cfg["autoregressive_mix"]
    if ar_mix > 0:
        ar = autoregressive_joint(history, cfg)
        model = (1.0 - ar_mix) * base + ar_mix * ar
    else:
        model = base
    lam = cfg["uniform_blend"]
    joint = lam * model + (1.0 - lam) * 0.001
    joint = np.clip(joint, EPS, None)
    return joint / joint.sum()


def exact_nll(actual: np.ndarray, joint: np.ndarray) -> float:
    idx = int(actual[0]) * 100 + int(actual[1]) * 10 + int(actual[2])
    return -math.log(max(float(joint[idx]), EPS))


def rank_of(actual: np.ndarray, joint: np.ndarray) -> int:
    idx = int(actual[0]) * 100 + int(actual[1]) * 10 + int(actual[2])
    ap = float(joint[idx])
    return 1 + int(np.sum(joint > ap))


def top_numbers(joint: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
    k = min(1000, int(k))
    idx = np.argpartition(joint, -k)[-k:]
    idx = idx[np.argsort(joint[idx])[::-1]]
    return [(f"{int(i):03d}", float(joint[i])) for i in idx]


def metrics_from_joint_rows(y: np.ndarray, joints: np.ndarray) -> dict[str, Any]:
    n = len(y)
    nlls = np.empty(n, dtype=float)
    ranks = np.empty(n, dtype=int)
    h1 = h20 = 0
    digit_losses = []
    for i in range(n):
        nlls[i] = exact_nll(y[i], joints[i])
        ranks[i] = rank_of(y[i], joints[i])
        idx = int(y[i, 0]) * 100 + int(y[i, 1]) * 10 + int(y[i, 2])
        order20 = np.argpartition(joints[i], -20)[-20:]
        h20 += int(idx in order20)
        h1 += int(idx == int(np.argmax(joints[i])))
        marg = joint_marginals(joints[i])
        digit_losses.append(np.mean([-math.log(max(float(marg[p, int(y[i, p])]), EPS)) for p in range(3)]))
    return {
        "n": n,
        "exact_nll": float(nlls.mean()),
        "digit_nll": float(np.mean(digit_losses)),
        "top1_rate": h1 / n,
        "top20_rate": h20 / n,
        "mean_rank": float(ranks.mean()),
        "score_exact_nlls": nlls.tolist(),
    }
