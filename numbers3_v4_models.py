#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

import numpy as np

from numbers3_champion import config_id
import numbers3_v3_models as v3

V4_DEFAULTS = {
    "power_beta": 1.0,
}

# v6 structural keys are deliberately optional. Old Champion configs do not gain
# these keys, so their config IDs and probability behaviour remain unchanged.
V6_OPTIONAL_DEFAULTS = {
    "interaction_mix": 0.0,
    "interaction_alpha": 2.0,
    "regime_mix": 0.0,
    "regime_window": 300,
}


def extended_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = v3.extended_config(config)
    for k, val in V4_DEFAULTS.items():
        cfg.setdefault(k, val)
    cfg["power_beta"] = float(cfg["power_beta"])
    if not (0.20 <= cfg["power_beta"] <= 2.00):
        raise ValueError("power_beta must be in [0.20, 2.00]")

    # Only materialise the v6 defaults after a config explicitly enters the v6
    # structural family. This keeps every legacy v3/v4/v5 config stable.
    if any(k in cfg for k in V6_OPTIONAL_DEFAULTS):
        for k, val in V6_OPTIONAL_DEFAULTS.items():
            cfg.setdefault(k, val)
        cfg["interaction_mix"] = float(cfg["interaction_mix"])
        cfg["interaction_alpha"] = float(cfg["interaction_alpha"])
        cfg["regime_mix"] = float(cfg["regime_mix"])
        cfg["regime_window"] = int(cfg["regime_window"])
        if not (0.0 <= cfg["interaction_mix"] <= 0.50):
            raise ValueError("interaction_mix must be in [0,0.50]")
        if cfg["interaction_alpha"] <= 0.0:
            raise ValueError("interaction_alpha must be > 0")
        if not (0.0 <= cfg["regime_mix"] <= 0.50):
            raise ValueError("regime_mix must be in [0,0.50]")
        if not (50 <= cfg["regime_window"] <= 3000):
            raise ValueError("regime_window must be in [50,3000]")

        # Canonicalise parameters that cannot affect the probability path. This
        # prevents the evolutionary search from spending trials on config IDs
        # that are numerically identical to another model.
        if cfg["interaction_mix"] <= 0.0 and cfg["regime_mix"] <= 0.0:
            for k in V6_OPTIONAL_DEFAULTS:
                cfg.pop(k, None)
        elif cfg["regime_mix"] <= 0.0:
            # regime_window is inactive for a pairwise-only model, so all such
            # configs share the same canonical value.
            cfg["regime_window"] = int(V6_OPTIONAL_DEFAULTS["regime_window"])
    return cfg


def v4_config_id(config: dict[str, Any]) -> str:
    return config_id(extended_config(config))


def power_calibrate(joint: np.ndarray, beta: float) -> np.ndarray:
    p = np.asarray(joint, dtype=float)
    p = np.clip(p, 1e-15, None)
    if abs(float(beta) - 1.0) < 1e-12:
        return p / p.sum()
    q = np.power(p, float(beta))
    q = np.clip(q, 1e-15, None)
    return q / q.sum()


def _valid_history(history: np.ndarray) -> np.ndarray:
    h = np.asarray(history, dtype=int)
    if h.ndim != 2 or h.shape[1] != 3:
        return np.empty((0, 3), dtype=int)
    return h[(h >= 0).all(axis=1) & (h <= 9).all(axis=1)]


def pairwise_interaction_joint(
    history: np.ndarray,
    config: dict[str, Any],
    *,
    window: int | None = None,
    half_life_override: float | None = None,
) -> np.ndarray:
    """Low-variance causal joint from all three within-number digit pairs.

    The three smoothed pair tables are converted to three chain factorizations
    and averaged. Compared with the full 10x10x10 autoregressive table, this
    estimates only pair relationships and therefore needs far less data.
    """
    cfg = extended_config(config)
    h = _valid_history(history)
    if window is not None and len(h) > int(window):
        h = h[-int(window):]
    if len(h) == 0:
        return np.full(1000, 0.001, dtype=float)

    alpha = float(cfg.get("interaction_alpha", 2.0))
    c01 = np.full((10, 10), alpha, dtype=float)
    c12 = np.full((10, 10), alpha, dtype=float)
    c02 = np.full((10, 10), alpha, dtype=float)

    half_life = float(
        half_life_override
        if half_life_override is not None
        else max(1, int(cfg.get("half_life_draws", 1500)))
    )
    age = (len(h) - 1) - np.arange(len(h), dtype=float)
    weights = np.power(0.5, age / max(half_life, 1.0))
    for row, w in zip(h, weights):
        a, b, c = map(int, row)
        c01[a, b] += w
        c12[b, c] += w
        c02[a, c] += w

    p01 = c01 / c01.sum()
    p12 = c12 / c12.sum()
    p02 = c02 / c02.sum()

    # P(a,b)P(c|b)
    p_c_given_b = c12 / c12.sum(axis=1, keepdims=True)
    j1 = p01[:, :, None] * p_c_given_b[None, :, :]

    # P(a,c)P(b|c)
    p_b_given_c = c12 / c12.sum(axis=0, keepdims=True)
    j2 = p02[:, None, :] * p_b_given_c[None, :, :]

    # P(b,c)P(a|b)
    p_a_given_b = c01 / c01.sum(axis=0, keepdims=True)
    j3 = p_a_given_b[:, :, None] * p12[None, :, :]

    joint = ((j1 + j2 + j3) / 3.0).reshape(-1)
    joint = np.clip(joint, 1e-15, None)
    return joint / joint.sum()


def regime_pairwise_joint(history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = extended_config(config)
    window = int(cfg.get("regime_window", 300))
    # A short half-life inside the window makes the component respond to a
    # genuine recent regime while smoothing strongly enough to avoid sparse
    # exact-number memorisation.
    return pairwise_interaction_joint(
        history,
        cfg,
        window=window,
        half_life_override=max(50.0, window / 2.0),
    )


def final_joint(base_position_probs: np.ndarray, history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = extended_config(config)

    # Reproduce the v3 joint exactly first. New structural components are mixed
    # before uniform shrinkage and power calibration, so the existing safety
    # floor remains effective.
    base = v3.position_joint(base_position_probs)
    ar_mix = float(cfg["autoregressive_mix"])
    if ar_mix > 0.0:
        ar = v3.autoregressive_joint(history, cfg)
        model = (1.0 - ar_mix) * base + ar_mix * ar
    else:
        model = base

    interaction_mix = float(cfg.get("interaction_mix", 0.0))
    if interaction_mix > 0.0:
        pairwise = pairwise_interaction_joint(history, cfg)
        model = (1.0 - interaction_mix) * model + interaction_mix * pairwise

    regime_mix = float(cfg.get("regime_mix", 0.0))
    if regime_mix > 0.0:
        recent = regime_pairwise_joint(history, cfg)
        model = (1.0 - regime_mix) * model + regime_mix * recent

    lam = float(cfg["uniform_blend"])
    joint = lam * model + (1.0 - lam) * 0.001
    joint = np.clip(joint, 1e-15, None)
    joint = joint / joint.sum()
    return power_calibrate(joint, cfg["power_beta"])


def build_features(nums: np.ndarray, config: dict[str, Any]):
    return v3.build_features(nums, extended_config(config))


def make_next_features(nums: np.ndarray, config: dict[str, Any]):
    return v3.make_next_features(nums, extended_config(config))


def fit_ml_models(X_train, y_train, config: dict[str, Any], seed: int | None = None):
    return v3.fit_ml_models(X_train, y_train, extended_config(config), seed=seed)


def frequency_probs(history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return v3.frequency_probs(history, extended_config(config))


def markov_probs(history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return v3.markov_probs(history, extended_config(config))


def top_numbers(joint: np.ndarray, k: int = 20):
    return v3.top_numbers(joint, k)


def joint_marginals(joint: np.ndarray) -> np.ndarray:
    return v3.joint_marginals(joint)


def metrics_from_joint_rows(y: np.ndarray, joints: np.ndarray):
    return v3.metrics_from_joint_rows(y, joints)


U_EXACT = math.log(1000.0)
