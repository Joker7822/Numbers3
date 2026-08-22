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


def extended_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = v3.extended_config(config)
    for k, val in V4_DEFAULTS.items():
        cfg.setdefault(k, val)
    cfg["power_beta"] = float(cfg["power_beta"])
    if not (0.20 <= cfg["power_beta"] <= 2.00):
        raise ValueError("power_beta must be in [0.20, 2.00]")
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


def final_joint(base_position_probs: np.ndarray, history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = extended_config(config)
    joint = v3.final_joint(base_position_probs, history, cfg)
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
