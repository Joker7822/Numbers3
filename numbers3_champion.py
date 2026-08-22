#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

DEFAULT_CONFIG = {
    "validation_draws": 1000,
    "top_k": 20,
    "half_life_draws": 1500,
    "evolution_rate": 0.70,
    "weight_temperature": 0.02,
    "frequency_profile": "default",
    "markov_blend": 0.65,
    "transition_window": 1000,
    "lr_c": 0.25,
    "tree_estimators": 240,
    "tree_min_leaf": 8,
    "random_state": 42,
}

FREQUENCY_PROFILES = {
    "default": [0.40, 0.30, 0.20, 0.10],
    "recent": [0.55, 0.25, 0.15, 0.05],
    "balanced": [0.25, 0.25, 0.25, 0.25],
    "medium": [0.20, 0.40, 0.30, 0.10],
}


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_CONFIG)
    if config:
        out.update(config)
    out["validation_draws"] = int(out["validation_draws"])
    out["top_k"] = int(out["top_k"])
    out["half_life_draws"] = int(out["half_life_draws"])
    out["evolution_rate"] = float(out["evolution_rate"])
    out["weight_temperature"] = float(out["weight_temperature"])
    out["markov_blend"] = float(out["markov_blend"])
    out["transition_window"] = int(out["transition_window"])
    out["lr_c"] = float(out["lr_c"])
    out["tree_estimators"] = int(out["tree_estimators"])
    out["tree_min_leaf"] = int(out["tree_min_leaf"])
    out["random_state"] = int(out["random_state"])
    if out["frequency_profile"] not in FREQUENCY_PROFILES:
        raise ValueError(f"unknown frequency_profile={out['frequency_profile']}")
    return out


def config_id(config: dict[str, Any]) -> str:
    payload = json.dumps(normalize_config(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_champion(path: Path) -> dict[str, Any]:
    if not path.exists():
        cfg = normalize_config(None)
        return {"champion_id": f"champ-{config_id(cfg)}", "generation": 1, "config": cfg}
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["config"] = normalize_config(raw.get("config"))
    raw.setdefault("champion_id", f"champ-{config_id(raw['config'])}")
    raw.setdefault("generation", 1)
    return raw


def frequency_probs(agent, history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = normalize_config(config)
    ws = np.asarray(FREQUENCY_PROFILES[cfg["frequency_profile"]], dtype=float)
    ws /= ws.sum()
    out = np.zeros((3, 10), dtype=float)
    for pos in range(3):
        for a, w in zip(ws, agent.WINDOWS):
            out[pos] += a * agent.dirichlet_freq(history, pos, w)
    return agent.normalize_probs(out)


def markov_probs(agent, history: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cfg = normalize_config(config)
    base = frequency_probs(agent, history, cfg)
    out = np.zeros((3, 10), dtype=float)
    if len(history) < 3:
        return base
    tw = cfg["transition_window"]
    blend = cfg["markov_blend"]
    h = history[-(tw + 1):] if len(history) > tw + 1 else history
    for pos in range(3):
        prev = int(history[-1, pos])
        from_d = h[:-1, pos]
        to_d = h[1:, pos]
        counts = np.bincount(to_d[from_d == prev], minlength=10).astype(float)
        conditional = agent.normalize_probs(counts + 1.0)
        out[pos] = blend * conditional + (1.0 - blend) * base[pos]
    return agent.normalize_probs(out)


def fit_ml_models(agent, X_train, y_train, config: dict[str, Any], seed: int | None = None):
    cfg = normalize_config(config)
    rs = cfg["random_state"] if seed is None else int(seed)
    sw = agent.recency_sample_weight(len(X_train), cfg["half_life_draws"])
    lr_models, et_models = [], []
    for pos in range(3):
        lr = LogisticRegression(
            C=cfg["lr_c"], max_iter=700, solver="lbfgs", random_state=rs + pos
        )
        lr.fit(X_train, y_train[:, pos], sample_weight=sw)
        lr_models.append(lr)
        et = ExtraTreesClassifier(
            n_estimators=cfg["tree_estimators"],
            min_samples_leaf=cfg["tree_min_leaf"],
            max_features="sqrt", n_jobs=-1, random_state=rs + 100 + pos,
        )
        et.fit(X_train, y_train[:, pos], sample_weight=sw)
        et_models.append(et)
    return {"logreg": lr_models, "extra_trees": et_models}


def apply_to_agent(agent, config: dict[str, Any]) -> dict[str, Any]:
    cfg = normalize_config(config)
    agent.frequency_probs = lambda history: frequency_probs(agent, history, cfg)
    agent.markov_probs = lambda history, trans_window=None: markov_probs(agent, history, cfg)
    agent.fit_ml_models = lambda X_train, y_train, agent_cfg: fit_ml_models(agent, X_train, y_train, cfg)
    return cfg
