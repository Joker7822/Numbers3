#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REPORT_VERSION = "replay-text-v1"


def _straight_bucket(row: pd.Series) -> str:
    if int(row.get("hit1", 0)):
        return "TOP1"
    if int(row.get("hit3", 0)):
        return "TOP3"
    if int(row.get("hit5", 0)):
        return "TOP5"
    if int(row.get("hit10", 0)):
        return "TOP10"
    if int(row.get("hit20", 0)):
        return "TOP20"
    return "OUT"


def _summary_line(name: str, m: dict[str, Any]) -> str:
    return (
        f"{name}: n={int(m['n'])} exact_nll={float(m['exact_nll']):.8f} "
        f"mean_rank={float(m['mean_rank']):.2f} top1={float(m['top1_rate']):.4%} "
        f"top20={float(m['top20_rate']):.4%} box20={float(m['box20_rate']):.4%} "
        f"mini20={float(m['mini20_rate']):.4%}"
    )


def _render(
    result: pd.DataFrame,
    champion: dict[str, Any],
    pool: dict[str, Any],
    replay: dict[str, Any],
    full: dict[str, Any],
    selection: dict[str, Any],
    audit: dict[str, Any],
    audit_draws: int,
) -> str:
    lines = [
        "NUMBERS3 最新モデル 全過去回再予測・照合結果",
        "=" * 112,
        f"report_version={REPORT_VERSION}",
        f"generated_at={replay['generated_at']}",
        f"primary_champion={champion['champion_id']} generation={int(champion.get('generation', 1))}",
        f"champion_pool={pool['pool_id']} members={len(pool.get('members') or [])}",
        f"latest_issue={int(replay['latest_issue'])} evaluated_draws={len(result)} audit_draws={audit_draws}",
        "method=causal_walk_forward (第N回の予測には第N-1回以前のデータだけを使用)",
        "note=これは最新モデルを過去へ再適用した再予測結果であり、当時保存されたlive予測そのものではありません。",
        "",
        _summary_line("SELECTION", selection),
        _summary_line("AUDIT_REPORT_ONLY", audit),
        _summary_line("FULL", full),
        "",
        "Pool members",
        "-" * 112,
    ]
    for member in pool.get("members") or []:
        lines.append(
            f"{member['champion_id']} cfg={member['config_id']} "
            f"weight={float(member.get('weight', 0.0)):.6%} dev_nll={member.get('dev_exact_nll')}"
        )

    lines += [
        "",
        "各回照合",
        "-" * 112,
        "columns: issue date segment pred actual rank straight box20 mini20 exact_nll refit_issue",
        "straight: TOP1/TOP3/TOP5/TOP10/TOP20/OUT; box20,mini20: 1=Top20内で該当 0=非該当",
    ]

    split = max(0, len(result) - int(audit_draws))
    for idx, row in result.reset_index(drop=True).iterrows():
        segment = "SEL" if idx < split else "AUDIT"
        lines.append(
            f"{int(row['target_issue']):07d} {row['draw_date']} {segment:5s} "
            f"pred={str(row['top1']).zfill(3)} actual={str(row['actual']).zfill(3)} "
            f"rank={int(row['actual_rank']):04d} straight={_straight_bucket(row):5s} "
            f"box20={int(row['box20'])} mini20={int(row['mini20'])} "
            f"nll={float(row['exact_nll']):.8f} refit={int(row['refit_issue']):07d}"
        )

    lines += [
        "",
        "照合件数",
        "-" * 112,
        f"Top1_hits={int(result['hit1'].sum())}",
        f"Top3_hits={int(result['hit3'].sum())}",
        f"Top5_hits={int(result['hit5'].sum())}",
        f"Top10_hits={int(result['hit10'].sum())}",
        f"Top20_hits={int(result['hit20'].sum())}",
        f"BOX20_hits={int(result['box20'].sum())}",
        f"MINI20_hits={int(result['mini20'].sum())}",
        "",
        "重要: AUDIT_REPORT_ONLYは探索・昇格・Pool重み決定には使用しません。",
    ]
    return "\n".join(lines) + "\n"


def write_replay_text_reports(
    outdir: Path,
    result: pd.DataFrame,
    champion: dict[str, Any],
    pool: dict[str, Any],
    replay: dict[str, Any],
    full: dict[str, Any],
    selection: dict[str, Any],
    audit: dict[str, Any],
    audit_draws: int,
) -> dict[str, Any]:
    text = _render(result, champion, pool, replay, full, selection, audit, audit_draws)
    outdir.mkdir(parents=True, exist_ok=True)
    latest = outdir / "latest_results.txt"
    latest.write_text(text, encoding="utf-8")

    archive_dir = outdir / "text_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    generation = int(champion.get("generation", 1))
    pool_id = str(pool.get("pool_id", "pool-unknown")).replace("/", "_")
    archive = archive_dir / f"generation_{generation:04d}_{pool_id}.txt"
    archive.write_text(text, encoding="utf-8")

    return {
        "version": REPORT_VERSION,
        "latest_path": str(latest),
        "archive_path": str(archive),
        "rows": int(len(result)),
    }
