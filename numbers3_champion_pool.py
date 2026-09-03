#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from numbers3_v4_models import extended_config, v4_config_id

JST = ZoneInfo("Asia/Tokyo")
POOL_VERSION = "champion-pool-v5-diversity"
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


def _now() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _member(champion_id: str, generation: int, config: dict[str, Any], dev_exact_nll: float | None,
            promoted_at: str | None = None) -> dict[str, Any]:
    cfg = extended_config(config)
    return {
        "champion_id": str(champion_id),
        "generation": int(generation),
        "config_id": v4_config_id(cfg),
        "config": cfg,
        "dev_exact_nll": None if dev_exact_nll is None else float(dev_exact_nll),
        "promoted_at": promoted_at or _now(),
        "weight": 0.0,
    }


def _weights(members: list[dict[str, Any]], temperature: float = 0.010) -> list[dict[str, Any]]:
    if not members:
        return members
    vals = np.array([
        float(m["dev_exact_nll"]) if m.get("dev_exact_nll") is not None else float("nan")
        for m in members
    ], dtype=float)
    if np.isfinite(vals).any():
        worst = float(np.nanmax(vals[np.isfinite(vals)])) + 0.02
        vals = np.where(np.isfinite(vals), vals, worst)
        best = float(vals.min())
        raw = np.exp(-(vals - best) / max(float(temperature), 1e-6))
    else:
        raw = np.ones(len(members), dtype=float)
    raw = raw / raw.sum()
    for m, w in zip(members, raw):
        m["weight"] = float(w)
    return members


def _structural_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ca = extended_config(a)
    cb = extended_config(b)
    different = 0
    for key in STRUCTURAL_KEYS:
        av = ca.get(key, 0.0 if key.endswith("_mix") else None)
        bv = cb.get(key, 0.0 if key.endswith("_mix") else None)
        if isinstance(av, float) or isinstance(bv, float):
            try:
                same = abs(float(av) - float(bv)) <= 1e-12
            except (TypeError, ValueError):
                same = av == bv
        else:
            same = av == bv
        different += int(not same)
    return different / max(1, len(STRUCTURAL_KEYS))


def _select_diverse_members(
    members: list[dict[str, Any]],
    primary_champion_id: str,
    max_members: int,
    min_distance: float = 0.20,
) -> list[dict[str, Any]]:
    if not members:
        return []

    # Deduplicate config IDs and keep the best measured version of each config.
    by_cfg: dict[str, dict[str, Any]] = {}
    for member in members:
        cid = member.get("config_id") or v4_config_id(member["config"])
        current = by_cfg.get(cid)
        m_nll = float("inf") if member.get("dev_exact_nll") is None else float(member["dev_exact_nll"])
        c_nll = float("inf") if current is None or current.get("dev_exact_nll") is None else float(current["dev_exact_nll"])
        if current is None or m_nll < c_nll:
            by_cfg[cid] = member

    candidates = sorted(
        by_cfg.values(),
        key=lambda m: (
            float("inf") if m.get("dev_exact_nll") is None else float(m["dev_exact_nll"]),
            -int(m.get("generation", 0)),
        ),
    )

    primary = next((m for m in candidates if m.get("champion_id") == primary_champion_id), None)
    if primary is None:
        primary = candidates[0]
    selected = [primary]
    remaining = [m for m in candidates if m is not primary]

    # Prefer promoted Champions with genuinely different structural assumptions.
    while remaining and len(selected) < max_members:
        eligible = [
            m for m in remaining
            if min(_structural_distance(m["config"], s["config"]) for s in selected) >= min_distance
        ]
        if eligible:
            pick = eligible[0]  # candidates are already ordered by dev NLL.
        else:
            # If history contains no sufficiently diverse promoted Champion, keep
            # the best remaining one rather than silently shrinking the Pool.
            pick = remaining[0]
        selected.append(pick)
        remaining.remove(pick)

    return selected


def pool_id(members: list[dict[str, Any]]) -> str:
    payload = [
        {"config_id": m.get("config_id") or v4_config_id(m["config"]), "weight": round(float(m.get("weight", 0.0)), 10)}
        for m in members
    ]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "pool-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def _pool_payload(primary_id: str, members: list[dict[str, Any]], updated_at: str | None = None) -> dict[str, Any]:
    return {
        "version": POOL_VERSION,
        "updated_at": updated_at or _now(),
        "primary_champion_id": primary_id,
        "diversity_policy": "promoted_champions_only; prefer >=20% structural-key distance; rejected models stay shadow-only",
        "members": members,
        "pool_id": pool_id(members),
    }


def load_pool(path: Path, champion: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            members = raw.get("members") or []
            clean = []
            for m in members:
                clean.append(_member(
                    m.get("champion_id", champion["champion_id"]),
                    int(m.get("generation", champion.get("generation", 1))),
                    m.get("config") or champion["config"],
                    m.get("dev_exact_nll"), m.get("promoted_at"),
                ))
            if clean:
                clean = _select_diverse_members(clean, champion["champion_id"], max_members=5)
                clean = _weights(clean)
                return _pool_payload(champion["champion_id"], clean, raw.get("updated_at") or _now())
        except Exception:
            pass
    current_nll = None
    nested = champion.get("nested_metrics_v4") or champion.get("nested_metrics_v3") or {}
    try:
        current_nll = float(nested["seed42"]["development"]["exact_nll"])
    except Exception:
        current_nll = None
    members = _weights([_member(champion["champion_id"], champion.get("generation", 1), champion["config"], current_nll,
                               champion.get("promoted_at"))])
    return _pool_payload(champion["champion_id"], members)


def seed_from_history(pool: dict[str, Any], history: pd.DataFrame, max_members: int = 5) -> dict[str, Any]:
    members = list(pool.get("members") or [])
    seen = {m.get("config_id") for m in members}
    if not history.empty and "decision" in history and "challenger_config_json" in history:
        promoted = history[history["decision"].astype(str).eq("PROMOTED")].copy()
        if "eval_version" in promoted:
            promoted = promoted[promoted["eval_version"].astype(str).str.contains(r"rolling(?:5|7)", na=False, regex=True)]
        for _, row in promoted.tail(max_members * 8).iterrows():
            try:
                cfg = extended_config(json.loads(str(row["challenger_config_json"])))
                cid = v4_config_id(cfg)
                if cid in seen:
                    continue
                nll = pd.to_numeric(pd.Series([row.get("dev_challenger_exact_nll")]), errors="coerce").iloc[0]
                members.append(_member(
                    str(row.get("champion_after") or f"champ-{cid}"),
                    0,
                    cfg, None if pd.isna(nll) else float(nll), str(row.get("generated_at") or _now()),
                ))
                seen.add(cid)
            except Exception:
                continue
    primary_id = str(pool.get("primary_champion_id") or (members[0]["champion_id"] if members else ""))
    members = _select_diverse_members(members, primary_id, max_members=max_members)
    members = _weights(members)
    pool.update(_pool_payload(primary_id, members))
    return pool


def update_pool(pool: dict[str, Any], previous_champion: dict[str, Any], new_champion: dict[str, Any],
                previous_dev_nll: float, new_dev_nll: float, max_members: int = 5) -> dict[str, Any]:
    existing = list(pool.get("members") or [])
    by_id = {m.get("config_id") or v4_config_id(m["config"]): m for m in existing}
    prev = _member(previous_champion["champion_id"], previous_champion.get("generation", 1), previous_champion["config"],
                   previous_dev_nll, previous_champion.get("promoted_at"))
    new = _member(new_champion["champion_id"], new_champion.get("generation", 1), new_champion["config"],
                  new_dev_nll, new_champion.get("promoted_at"))
    by_id[prev["config_id"]] = prev
    by_id[new["config_id"]] = new
    members = _select_diverse_members(list(by_id.values()), new_champion["champion_id"], max_members=max_members)
    members = _weights(members)
    return _pool_payload(new_champion["champion_id"], members)


def save_pool(path: Path, pool: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
