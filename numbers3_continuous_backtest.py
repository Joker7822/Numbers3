#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one leak-free historical backtest cycle and persist continuous-run telemetry."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

JST = ZoneInfo("Asia/Tokyo")
HISTORY_COLUMNS = [
    "cycle",
    "generated_at",
    "dataset_fingerprint",
    "latest_issue",
    "evaluated_draws",
    "duration_seconds",
    "top1_hits",
    "top1_rate",
    "top20_hits",
    "top20_rate",
    "exact_nll",
    "uniform_exact_nll",
    "nll_improvement_vs_uniform",
]


def _match(pattern: str, text: str, label: str):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"continuous backtest: {label} を解析できません")
    return m


def parse_summary(text: str) -> dict:
    top1 = _match(r"^Top1\s*:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%", text, "Top1")
    top20 = _match(r"^Top20\s*:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%", text, "Top20")
    nll = _match(
        r"Exact NLL平均:\s*([0-9.]+)\s*/\s*一様:\s*([0-9.]+)\s*/\s*差:\s*([+-]?[0-9.]+)",
        text,
        "Exact NLL",
    )
    return {
        "top1_hits": int(top1.group(1)),
        "evaluated_draws": int(top1.group(2)),
        "top1_rate": float(top1.group(3)) / 100.0,
        "top20_hits": int(top20.group(1)),
        "top20_rate": float(top20.group(3)) / 100.0,
        "exact_nll": float(nll.group(1)),
        "uniform_exact_nll": float(nll.group(2)),
        "nll_improvement_vs_uniform": float(nll.group(3)),
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def append_history(path: Path, row: dict, max_rows: int) -> None:
    if path.exists():
        hist = pd.read_csv(path)
    else:
        hist = pd.DataFrame(columns=HISTORY_COLUMNS)
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    for col in HISTORY_COLUMNS:
        if col not in hist.columns:
            hist[col] = ""
    if len(hist) > max_rows:
        hist = hist.tail(max_rows).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist[HISTORY_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")


def render_status(path: Path, state: dict, row: dict) -> None:
    edge = float(row["nll_improvement_vs_uniform"])
    lines = [
        "NUMBERS3 連続バックテスト状態",
        "=" * 72,
        f"更新日時: {row['generated_at']}",
        f"累計サイクル: {int(state['total_cycles']):,}",
        f"最新サイクル: {int(row['cycle']):,}",
        f"対象最新回: 第{int(row['latest_issue'])}回",
        f"評価回数: {int(row['evaluated_draws']):,}回",
        f"1サイクル所要時間: {float(row['duration_seconds']):.1f}秒",
        "",
        "最新サイクル結果",
        "-" * 72,
        f"Top1 : {int(row['top1_hits'])}/{int(row['evaluated_draws'])} = {float(row['top1_rate']):.4%}",
        f"Top20: {int(row['top20_hits'])}/{int(row['evaluated_draws'])} = {float(row['top20_rate']):.4%}",
        f"Exact NLL: {float(row['exact_nll']):.8f}",
        f"Uniform NLL: {float(row['uniform_exact_nll']):.8f}",
        f"差(一様-model): {edge:+.8f}",
        f"優位性判定: {'positive' if edge > 0 else 'not_confirmed'}",
        "",
        "運転方式",
        "-" * 72,
        "・1サイクル完了後、待機せず直ちに次サイクルを開始します。",
        "・同一データ/同一コードでの再実行は独立した統計標本ではなく、再現性確認です。",
        "・新しい抽せんデータまたはコード変更を取り込むと、次サイクルから再評価します。",
        "・GitHub-hosted runnerの上限前に次のworkflow runへ引き継ぎます。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one continuous NUMBERS3 backtest cycle")
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--backtest-script", default="numbers3_historical_backtest.py")
    ap.add_argument("--history", default="backtests/continuous_backtest_history.csv")
    ap.add_argument("--status", default="backtests/continuous_status.txt")
    ap.add_argument("--state", default="state/continuous_backtest_state.json")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refit-interval", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-history-rows", type=int, default=10000)
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    script_path = Path(args.backtest_script).resolve()
    history_path = Path(args.history)
    status_path = Path(args.status)
    state_path = Path(args.state)

    state = load_state(state_path)
    cycle = int(state.get("total_cycles", 0)) + 1
    started = time.monotonic()
    tmpdir = Path(tempfile.mkdtemp(prefix="numbers3_continuous_"))
    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--csv", str(csv_path),
            "--output-dir", str(tmpdir),
            "--warmup", str(args.warmup),
            "--refit-interval", str(args.refit_interval),
            "--topk", str(args.topk),
            "--force",
        ]
        subprocess.run(cmd, check=True)
        meta = json.loads((tmpdir / "historical_meta.json").read_text(encoding="utf-8"))
        summary_text = (tmpdir / "historical_accuracy.txt").read_text(encoding="utf-8")
        parsed = parse_summary(summary_text)
    finally:
        duration = time.monotonic() - started
        shutil.rmtree(tmpdir, ignore_errors=True)

    now = datetime.now(JST).isoformat(timespec="seconds")
    row = {
        "cycle": cycle,
        "generated_at": now,
        "dataset_fingerprint": meta["dataset_fingerprint"],
        "latest_issue": int(meta["latest_issue"]),
        "evaluated_draws": int(parsed["evaluated_draws"]),
        "duration_seconds": round(duration, 3),
        "top1_hits": int(parsed["top1_hits"]),
        "top1_rate": parsed["top1_rate"],
        "top20_hits": int(parsed["top20_hits"]),
        "top20_rate": parsed["top20_rate"],
        "exact_nll": parsed["exact_nll"],
        "uniform_exact_nll": parsed["uniform_exact_nll"],
        "nll_improvement_vs_uniform": parsed["nll_improvement_vs_uniform"],
    }

    append_history(history_path, row, args.max_history_rows)
    state = {
        "total_cycles": cycle,
        "last_completed_at": now,
        "last_dataset_fingerprint": meta["dataset_fingerprint"],
        "last_issue": int(meta["latest_issue"]),
        "last_duration_seconds": round(duration, 3),
        "last_exact_nll": parsed["exact_nll"],
        "last_top1_rate": parsed["top1_rate"],
        "last_top20_rate": parsed["top20_rate"],
        "history_rows_kept": min(cycle, args.max_history_rows),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    render_status(status_path, state, row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
