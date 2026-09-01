#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-15
MODE_LABELS = {
    "straight": "Straight",
    "box": "BOX",
    "set": "SET",
    "mini": "MINI",
}


def issue_no(v) -> int:
    m = re.search(r"(\d+)", str(v))
    if not m:
        raise ValueError(f"cannot parse issue: {v!r}")
    return int(m.group(1))


def robust_number3(v) -> str:
    if pd.isna(v):
        return ""
    raw = str(v).strip()
    try:
        n = int(float(raw.replace(",", "")))
        if 0 <= n <= 999:
            return f"{n:03d}"
    except Exception:
        pass
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits[:3] if len(digits) >= 3 else ""


def probability_vector(raw) -> np.ndarray | None:
    try:
        p = np.asarray(json.loads(raw), dtype=float)
    except Exception:
        return None
    if p.shape != (1000,) or not np.all(np.isfinite(p)):
        return None
    p = np.clip(p, EPS, None)
    p /= p.sum()
    return p


def forecast_hash(row: pd.Series) -> str:
    payload = (
        f"{row.get('target_issue', '')}|{row.get('generated_at', '')}|"
        f"{row.get('champion_id', '')}|{row.get('config_id', '')}|"
        f"{row.get('probabilities_json', '')}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def money(v) -> str:
    if pd.isna(v):
        return "算出不可"
    return f"{int(round(float(v))):,}円"


def signed_money(v) -> str:
    if pd.isna(v):
        return "算出不可"
    return f"{int(round(float(v))):+,}円"


def mode_totals(score: pd.DataFrame, mode: str) -> tuple[float, float, float, float]:
    stake = float(pd.to_numeric(score[f"{mode}_stake"], errors="coerce").fillna(0).sum())
    returns = pd.to_numeric(score[f"{mode}_return"], errors="coerce")
    if returns.isna().any():
        return stake, np.nan, np.nan, np.nan
    ret = float(returns.sum())
    profit = ret - stake
    roi = profit / stake if stake else np.nan
    return stake, ret, profit, roi


def archive_meta(archive: pd.DataFrame, target: int, generated_at: str) -> tuple[int, str]:
    work = archive[pd.to_numeric(archive["target_issue"], errors="coerce").eq(target)].copy()
    versions = len(work)
    if work.empty:
        return 0, ""
    exact = work[work["generated_at"].astype(str).eq(str(generated_at))]
    row = exact.iloc[-1] if not exact.empty else work.iloc[-1]
    return versions, forecast_hash(row)


def write_purchase_summary(
    score_path: Path,
    archive_path: Path,
    out_txt: Path,
    missing: list[int],
) -> None:
    score = pd.read_csv(score_path, dtype={"actual": str}) if score_path.exists() else pd.DataFrame()
    archive = pd.read_csv(archive_path, dtype={"probabilities_json": str}) if archive_path.exists() else pd.DataFrame()

    lines = [
        "NUMBERS3 通常予測 購入方式別Live照合",
        "=" * 112,
        "正式基準: 各回について販売締切18:30 JSTより前に保存された最後の通常予測を採用。",
        "Champion replayではなく、当時保存された1000通り確率だけを採点。",
        "仮想購入: Top20から重複する同一賭式を除き、Straight / BOX / SET / MINIを各1口。1口=200円。",
        "全口外れはこのTXTの各回一覧には掲載しないが、CSV・統計母数・ROIには残す。",
        "",
    ]

    if score.empty:
        lines.append("照合済み通常予測はありません。")
    else:
        n = len(score)
        exact = pd.to_numeric(score["exact_nll"], errors="coerce")
        ranks = pd.to_numeric(score["actual_rank"], errors="coerce")
        any_hit = pd.to_numeric(score["any_purchase_hit"], errors="coerce").fillna(0)
        total_stake = float(pd.to_numeric(score["total_stake"], errors="coerce").fillna(0).sum())
        total_returns = pd.to_numeric(score["total_return"], errors="coerce")
        complete = not total_returns.isna().any()
        total_return = float(total_returns.sum()) if complete else np.nan
        total_profit = total_return - total_stake if complete else np.nan
        total_roi = total_profit / total_stake if complete and total_stake else np.nan

        lines += [
            "総合",
            "-" * 112,
            f"採点済み回数: {n}",
            f"Exact NLL: {exact.mean():.8f} / 一様={math.log(1000):.8f}",
            f"平均実数字順位: {ranks.mean():.2f}/1000",
            f"Straight Top1: {pd.to_numeric(score['top1_hit'], errors='coerce').mean():.4%} / "
            f"Top20: {pd.to_numeric(score['top20_straight_hit'], errors='coerce').mean():.4%}",
            f"購入方式で何らか的中: {int(any_hit.sum())}/{n} ({any_hit.mean():.4%})",
            f"仮想購入額合計: {money(total_stake)}",
            f"払戻合計: {money(total_return)}",
            f"収支: {signed_money(total_profit)}",
            f"合算ROI: {total_roi:+.4%}" if pd.notna(total_roi) else "合算ROI: 算出不可",
            f"全口外れ非掲載回数: {int((any_hit == 0).sum())}",
            "",
            "方式別成績",
            "-" * 112,
        ]

        for mode in ("straight", "box", "set", "mini"):
            stake, ret, profit, roi = mode_totals(score, mode)
            hit_rate = pd.to_numeric(score[f"{mode}_hit"], errors="coerce").mean()
            roi_text = f"{roi:+.4%}" if pd.notna(roi) else "算出不可"
            lines.append(
                f"{MODE_LABELS[mode]:8s} hit={hit_rate:.4%} stake={money(stake)} "
                f"return={money(ret)} profit={signed_money(profit)} ROI={roi_text}"
            )

        lines += ["", "的中回のみ", "-" * 112]
        hits = score[any_hit.eq(1)].copy()
        if hits.empty:
            lines.append("該当なし")
        else:
            for _, r in hits.iterrows():
                target = int(r["target_issue"])
                versions, fh = archive_meta(archive, target, str(r.get("generated_at", "")))
                roi = pd.to_numeric(pd.Series([r.get("roi")]), errors="coerce").iloc[0]
                roi_text = f"{roi:+.2%}" if pd.notna(roi) else "算出不可"
                lines.append(
                    f"第{target}回 {r['draw_date']} actual={str(r['actual']).zfill(3)} "
                    f"rank={int(r['actual_rank'])} hits={r.get('hit_modes', '') or '-'} "
                    f"stake={money(r['total_stake'])} return={money(r['total_return'])} "
                    f"profit={signed_money(r['profit'])} ROI={roi_text} "
                    f"NLL={float(r['exact_nll']):.8f} archived_versions={versions} hash={fh}"
                )
                lines.append(f"  {r.get('hit_details', '')}")

    lines += ["", "予測欠番監査", "-" * 112]
    if missing:
        lines.append("WARNING: 抽せん結果はあるが、締切前の通常予測アーカイブが無い回があります。")
        lines.append("missing_issues=" + ",".join(str(x) for x in missing))
    else:
        lines.append("missing_issues=none")

    lines += [
        "",
        "注意: 仮想購入は成績監査用の固定ポリシーであり、実際に購入したことを意味しません。",
        "正式な通常予測成績はこのファイルと live_purchase_scoreboard.csv を基準にします。",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest_pending(archive: pd.DataFrame, drawn: set[int]) -> tuple[pd.Series, np.ndarray] | None:
    if archive.empty:
        return None
    work = archive.copy()
    work["_target"] = pd.to_numeric(work["target_issue"], errors="coerce")
    work["_generated"] = pd.to_datetime(work["generated_at"], errors="coerce", utc=True)
    work = work[work["_target"].notna()]
    work = work[~work["_target"].astype(int).isin(drawn)]
    work = work.sort_values(["_target", "_generated"], ascending=[False, False])
    for _, row in work.iterrows():
        p = probability_vector(row.get("probabilities_json", ""))
        if p is not None:
            return row, p
    return None


def write_canonical_cumulative(
    csv_path: Path,
    score_path: Path,
    archive_path: Path,
    out_path: Path,
    missing: list[int],
) -> None:
    draws = pd.read_csv(csv_path).copy()
    draws["_issue"] = draws["回別"].map(issue_no)
    draws["_actual"] = draws["本数字"].map(robust_number3)
    drawn = set(draws["_issue"].astype(int))

    score = pd.read_csv(score_path, dtype={"actual": str}) if score_path.exists() else pd.DataFrame()
    archive = pd.read_csv(archive_path, dtype={"probabilities_json": str}) if archive_path.exists() else pd.DataFrame()

    lines = [
        "NUMBERS3 通常予測・正式Live照合（canonical）",
        "=" * 104,
        "採用ルール: 販売締切18:30 JSTより前に保存された最後の予測のみ。",
        "同じ回に古い予測版と新しい予測版がある場合、新しい締切前予測を正式成績とします。",
        "照合済み全口外れは非掲載。ただし live_purchase_scoreboard.csv の統計には残します。",
        "",
    ]

    pending = latest_pending(archive, drawn)
    if pending is not None:
        row, p = pending
        target = int(row["_target"])
        order = np.argsort(p)[::-1][:20]
        lines += [
            f"第{target}回  状態=未抽せん  予測生成={row.get('generated_at', '')}",
            f"Champion={row.get('champion_id', '')} generation={row.get('generation', '')} "
            f"config={row.get('config_id', '')} hash={forecast_hash(row)}",
            "-" * 104,
        ]
        for i, idx in enumerate(order, 1):
            lines.append(f"{i:02d}. {int(idx):03d}  p={float(p[idx]):.6%}")
        lines.append("")

    lines += ["照合済み的中回", "-" * 104]
    if score.empty:
        lines.append("該当なし")
        omitted = 0
    else:
        any_hit = pd.to_numeric(score["any_purchase_hit"], errors="coerce").fillna(0)
        hits = score[any_hit.eq(1)].sort_values("target_issue", ascending=False)
        omitted = int((any_hit == 0).sum())
        if hits.empty:
            lines.append("該当なし")
        else:
            for _, r in hits.iterrows():
                target = int(r["target_issue"])
                versions, fh = archive_meta(archive, target, str(r.get("generated_at", "")))
                roi = pd.to_numeric(pd.Series([r.get("roi")]), errors="coerce").iloc[0]
                roi_text = f"{roi:+.2%}" if pd.notna(roi) else "算出不可"
                lines.append(
                    f"第{target}回 {r['draw_date']} actual={str(r['actual']).zfill(3)} "
                    f"generated={r['generated_at']} rank={int(r['actual_rank'])} "
                    f"hits={r.get('hit_modes', '') or '-'} NLL={float(r['exact_nll']):.8f}"
                )
                lines.append(
                    f"  top20={r['top20_numbers']} / stake={money(r['total_stake'])} "
                    f"return={money(r['total_return'])} profit={signed_money(r['profit'])} "
                    f"ROI={roi_text} / archived_versions={versions} hash={fh}"
                )
                lines.append(f"  {r.get('hit_details', '')}")

    lines += [
        "",
        f"照合済み全口外れ非掲載回数: {omitted}",
        "欠番: " + (",".join(str(x) for x in missing) if missing else "none"),
        "",
        "方式別ROI・全回統計は predictions/live_purchase_scoreboard.txt を参照。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_live_reports(
    csv_path: Path,
    archive_path: Path,
    score_path: Path,
    purchase_txt: Path,
    cumulative_txt: Path,
    missing: list[int],
) -> dict:
    write_purchase_summary(score_path, archive_path, purchase_txt, missing)
    write_canonical_cumulative(csv_path, score_path, archive_path, cumulative_txt, missing)
    return {
        "reporting_version": "canonical-live-v2",
        "purchase_txt": str(purchase_txt),
        "cumulative_txt": str(cumulative_txt),
    }
