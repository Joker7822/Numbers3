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
POOL_VERSION = "champion-pool-v4"


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


def pool_id(members: list[dict[str, Any]]) -> str:
    payload = [
        {"config_id": m.get("config_id") or v4_config_id(m["config"]), "weight": round(float(m.get("weight", 0.0)), 10)}
        for m in members
    ]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "pool-" + hashlib.sha256(text.encode()).hexdigest()[:12]


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
                clean = _weights(clean)
                return {
                    "version": POOL_VERSION,
                    "updated_at": raw.get("updated_at") or _now(),
                    "primary_champion_id": champion["champion_id"],
                    "members": clean,
                    "pool_id": pool_id(clean),
                }
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
    return {
        "version": POOL_VERSION, "updated_at": _now(),
        "primary_champion_id": champion["champion_id"], "members": members, "pool_id": pool_id(members),
    }


def seed_from_history(pool: dict[str, Any], history: pd.DataFrame, max_members: int = 5) -> dict[str, Any]:
    members = list(pool.get("members") or [])
    seen = {m.get("config_id") for m in members}
    if not history.empty and "decision" in history and "challenger_config_json" in history:
        promoted = history[history["decision"].astype(str).eq("PROMOTED")].copy()
        if "eval_version" in promoted:
            promoted = promoted[promoted["eval_version"].astype(str).str.contains("rolling5", na=False)]
        for _, row in promoted.tail(max_members * 3).iterrows():
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
    members = sorted(members, key=lambda m: (float("inf") if m.get("dev_exact_nll") is None else float(m["dev_exact_nll"]), -int(m.get("generation", 0))))[:max_members]
    members = _weights(members)
    pool["members"] = members
    pool["pool_id"] = pool_id(members)
    pool["updated_at"] = _now()
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
    members = sorted(by_id.values(), key=lambda m: (float("inf") if m.get("dev_exact_nll") is None else float(m["dev_exact_nll"]), -int(m.get("generation", 0))))[:max_members]
    members = _weights(members)
    return {
        "version": POOL_VERSION,
        "updated_at": _now(),
        "primary_champion_id": new_champion["champion_id"],
        "members": members,
        "pool_id": pool_id(members),
    }


def save_pool(path: Path, pool: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
