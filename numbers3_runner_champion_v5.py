#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import numbers3_runner as base_runner
import numbers3_runner_champion_v4 as v4_runner
from numbers3_live_purchase_audit import update_purchase_audit


def _install_hits_only_cumulative_renderer():
    original = base_runner.render_cumulative

    def render_hits_only(path: Path, history: pd.DataFrame):
        if history.empty:
            original(path, history)
            return

        work = history.copy()
        work["_target"] = pd.to_numeric(work["target_issue"], errors="coerce")
        keep_indices = []
        omitted_targets = 0
        for _, grp in work.groupby("_target", dropna=False):
            statuses = grp["status"].fillna("").astype(str)
            resolved = statuses.eq("照合済").all()
            matches = grp["match_types"].fillna("").astype(str).str.strip()
            has_hit = ((matches != "") & (matches != "はずれ")).any()
            if resolved and not has_hit:
                omitted_targets += 1
                continue
            keep_indices.extend(grp.index.tolist())

        filtered = history.loc[sorted(keep_indices)].copy() if keep_indices else history.iloc[0:0].copy()
        original(path, filtered)
        text = path.read_text(encoding="utf-8")
        header = (
            f"表示方針: 照合済みで全口外れの回は非掲載（統計・CSV履歴には残す）\n"
            f"非掲載回数: {omitted_targets}\n\n"
        )
        path.write_text(text.replace("\n\n", "\n" + header, 1), encoding="utf-8")

    base_runner.render_cumulative = render_hits_only


def main():
    raw_args = list(sys.argv[1:])
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--csv", default="data/numbers3.csv")
    p.add_argument("--pred-dir", default="predictions")
    known, _ = p.parse_known_args(raw_args)

    _install_hits_only_cumulative_renderer()
    v4_runner.main()

    pred_dir = Path(known.pred_dir)
    result = update_purchase_audit(
        Path(known.csv),
        pred_dir / "live_forecast_archive.csv",
        pred_dir / "live_purchase_scoreboard.csv",
        pred_dir / "live_purchase_scoreboard.txt",
    )
    (pred_dir / "live_purchase_audit_state.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"live_purchase_audit": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
