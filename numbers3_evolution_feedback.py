#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one Evolution Engine step with historical-replay feedback.

When the current Champion has stopped improving on the causal full-history development
replay, widen the Challenger mutations automatically. The final audit holdout is never
used for mutation or promotion decisions.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numbers3_evolution_engine as engine
from numbers3_champion import config_id, normalize_config

REPLAY_STATE = Path("state/champion_replay_state.json")


def replay_stagnation() -> int:
    try:
        raw = json.loads(REPLAY_STATE.read_text(encoding="utf-8"))
        return max(0, int(raw.get("stagnation_count", 0)))
    except Exception:
        return 0


def feedback_mutate(champion: dict, experiment_no: int, fp: str, seen: set[str]) -> dict:
    stagnation = replay_stagnation()
    if stagnation < 3:
        return ORIGINAL_MUTATE(champion, experiment_no, fp, seen)

    base = normalize_config(champion)
    seed_text = f"feedback:{fp}:{experiment_no}:{config_id(base)}:{stagnation}"
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    keys = list(engine.SEARCH_SPACE)

    # The longer the causal historical replay fails to set a new best, the broader
    # the search becomes. This changes search breadth only; promotion thresholds stay fixed.
    if stagnation >= 12:
        min_changes, max_changes = 4, min(7, len(keys))
    elif stagnation >= 6:
        min_changes, max_changes = 3, min(6, len(keys))
    else:
        min_changes, max_changes = 2, min(4, len(keys))

    for _ in range(200):
        candidate = dict(base)
        count = rng.randint(min_changes, max_changes)
        for key in rng.sample(keys, count):
            values = list(engine.SEARCH_SPACE[key])
            current = candidate[key]
            alternatives = [v for v in values if v != current]
            if alternatives:
                candidate[key] = rng.choice(alternatives)
        candidate = normalize_config(candidate)
        cid = config_id(candidate)
        if cid != config_id(base) and cid not in seen:
            return candidate

    return ORIGINAL_MUTATE(champion, experiment_no + 100003 * max(1, stagnation), fp, seen)


ORIGINAL_MUTATE = engine.mutate_config
engine.mutate_config = feedback_mutate

if __name__ == "__main__":
    engine.main()
