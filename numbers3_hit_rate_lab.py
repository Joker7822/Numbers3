#!/usr/bin/env python3
"""Pre-draw probability-only experimental BOX selection methods.

Experimental policies are SHADOW evaluations; the production BOX purchase
policy remains unchanged until credible, prospective evidence warrants it.
The historical winning number and payout are never inputs to these selectors.
"""
from __future__ import annotations

from itertools import combinations_with_replacement, permutations
import math
import numpy as np

# Every valid BOX digit multiset, sorted by its canonical 3-digit ticket.
_GROUPS = []
for comb in combinations_with_replacement(range(10), 3):
    if comb[0] == comb[2]:  # 000, 111, ... are not BOX tickets.
        continue
    key = "".join(str(d) for d in comb)
    indices = sorted({int("".join(map(str, p))) for p in permutations(comb)})
    _GROUPS.append((key, tuple(indices)))
BOX_KEYS = tuple(key for key, _ in _GROUPS)
BOX_GROUP_INDICES = tuple(np.asarray(ids, dtype=int) for _, ids in _GROUPS)
BOX_DISTINCT_MASK = np.array([len(set(k)) == 3 for k in BOX_KEYS], dtype=bool)
assert len(BOX_KEYS) == 210 and int(BOX_DISTINCT_MASK.sum()) == 120


def best_box_by_mass(joint, *, distinct_only: bool) -> tuple[str, float]:
    """Choose highest total permutation mass (not highest single permutation).

    distinct_only=True optimizes within six-permutation tickets, preserving a
    6/1000 uniform-null hit baseline. It does not promise a forecasting edge.
    """
    p = np.asarray(joint, dtype=float)
    if p.shape != (1000,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("expected a finite nonnegative 1000-number distribution")
    p = p / p.sum()
    group_mass = np.fromiter((p[ids].sum() for ids in BOX_GROUP_INDICES),
                             dtype=float, count=len(BOX_GROUP_INDICES))
    if distinct_only:
        group_mass[~BOX_DISTINCT_MASK] = -1.0
    best = int(np.argmax(group_mass))  # Stable lexicographic tie break.
    return BOX_KEYS[best], float(group_mass[best])


def box_hit(number: str, actual: str) -> int:
    return int("".join(sorted(str(number).zfill(3))) ==
               "".join(sorted(str(actual).zfill(3))))


def paired_exact_p(old_only: int, new_only: int) -> float:
    """Two-sided exact paired-sign/McNemar p-value; exploratory only."""
    if min(old_only, new_only) < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = old_only + new_only
    if n == 0:
        return 1.0
    k = min(old_only, new_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def shadow_comparison(df) -> dict:
    """Report hit counts, paired differences and distribution-shift period split.

    No strategy is promoted based on this retrospective comparison alone.
    """
    if df.empty:
        return {"available": False}
    required = ("box_ticket_hit", "box_mass_hit", "box_unique_hit")
    if any(c not in df.columns for c in required):
        return {"available": False}
    old = df["box_ticket_hit"].astype(int)
    policies = {
        "baseline": old,
        "mass": df["box_mass_hit"].astype(int),
        "unique": df["box_unique_hit"].astype(int),
    }
    out = {"available": True, "draws": int(len(df)), "retrospective_policy_design": True,
           "production_policy_unchanged": True, "modes": {}}
    for name, hits in policies.items():
        n_old_only = int(((old == 1) & (hits == 0)).sum())
        n_new_only = int(((old == 0) & (hits == 1)).sum())
        out["modes"][name] = {
            "hits": int(hits.sum()),
            "hit_rate": float(hits.mean()),
            "delta_hits_vs_baseline": int(hits.sum() - old.sum()),
            "baseline_only": n_old_only,
            "challenger_only": n_new_only,
            "paired_p_exploratory": paired_exact_p(n_old_only, n_new_only),
        }
    if "target_issue" in df:
        holdout = df[df["target_issue"] >= 5000]
        if not holdout.empty:
            out["recent_5000_plus"] = {
                "draws": int(len(holdout)),
                **{name + "_hits": int(holdout[{"baseline":"box_ticket_hit", "mass":"box_mass_hit","unique":"box_unique_hit"}[name]].sum())
                   for name in policies},
            }
    return out
