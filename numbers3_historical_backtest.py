#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NUMBERS3 past-only walk-forward accuracy audit."""
from __future__ import annotations

import argparse, hashlib, json, math, re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

from numbers3_evo_agent import (
    AgentConfig, MODEL_NAMES, build_features, ensemble_probs, frequency_probs,
    load_history, normalize_probs, predict_ml, top_k_numbers,
)

JST = ZoneInfo("Asia/Tokyo")
U_DIGIT = math.log(10.0)
U_EXACT = math.log(1000.0)
KS = (1, 3, 5, 10, 20)
MIN_ML_TRAIN_DRAWS = 100
TICKET_PRICE_YEN = 200


def issue_no(v):
    m = re.search(r"(\d+)", str(v))
    if not m:
        raise ValueError(f"回号を解釈できません: {v!r}")
    return int(m.group(1))


def num3(a):
    return "".join(str(int(x)) for x in a)


def digit_nll(actual, p):
    return float(np.mean([-math.log(max(float(p[i, int(actual[i])]), 1e-12)) for i in range(3)]))


def joint_probs(p):
    p = normalize_probs(np.asarray(p, float))
    return (p[0, :, None, None] * p[1, None, :, None] * p[2, None, None, :]).reshape(-1)


def adaptive_weights(losses, temp):
    best = min(losses.values())
    raw = {k: math.exp(-(v - best) / max(temp, 1e-6)) for k, v in losses.items()}
    s = sum(raw.values())
    return {k: raw[k] / s for k in MODEL_NAMES}


def fit_logreg(X, y, end, seed):
    idx = np.arange(5, end)
    out = []
    for pos in range(3):
        m = SGDClassifier(loss="log_loss", alpha=5e-4, max_iter=1000, tol=1e-3,
                          average=True, random_state=seed + pos)
        m.fit(X.iloc[idx], y[idx, pos])
        out.append(m)
    return out


def fit_trees(X, y, end, seed):
    idx = np.arange(5, end)
    out = []
    for pos in range(3):
        m = ExtraTreesClassifier(n_estimators=40, min_samples_leaf=10,
                                 max_features="sqrt", n_jobs=1,
                                 random_state=seed + pos)
        m.fit(X.iloc[idx], y[idx, pos])
        out.append(m)
    return out


def markov_probs(history, base, window=1000):
    if len(history) < 3:
        return base
    h = history[-(window + 1):]
    out = np.zeros((3, 10), float)
    for pos in range(3):
        prev = int(history[-1, pos])
        counts = np.bincount(h[1:, pos][h[:-1, pos] == prev], minlength=10).astype(float)
        cond = normalize_probs(counts + 1.0)
        out[pos] = 0.65 * cond + 0.35 * base[pos]
    return normalize_probs(out)


def fingerprint(df):
    cols = [c for c in ["抽せん日", "本数字", "回別", "ストレート", "ボックス",
                           "セット(ストレート)", "セット(ボックス)", "ミニ"] if c in df]
    text = df[cols].fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def wilson(h, n, z=1.959963984540054):
    if not n:
        return 0.0, 0.0
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0, c - m), min(1, c + m)


def metrics(df):
    n = len(df)
    out = {
        "n": n,
        "exact_nll": float(df.exact_nll.mean()),
        "digit_nll": float(df.digit_nll.mean()),
        "mean_rank": float(df.actual_rank.mean()),
        "median_rank": float(df.actual_rank.median()),
        "box20": int(df.box20.sum()),
        "mini20": int(df.mini20.sum()),
    }
    for k in KS:
        out[f"h{k}"] = int(df[f"hit{k}"].sum())
        out[f"r{k}"] = out[f"h{k}"] / n
    return out


def rate_line(k, h, n):
    r = h / n
    lo, hi = wilson(h, n)
    base = k / 1000
    return (f"Top{k:<2}: {h:>4}/{n} = {r:.4%}  95%CI=[{lo:.4%}, {hi:.4%}]  "
            f"一様={base:.3%}  ratio={r/base:.3f}x  一様基準CI内={'YES' if lo <= base <= hi else 'NO'}")


def write_summary(path, df, warmup, refit):
    m = metrics(df)
    n = m["n"]
    lo1, hi1 = wilson(m["h1"], n)
    lo20, hi20 = wilson(m["h20"], n)
    edge = U_EXACT - m["exact_nll"]
    confirmed = (lo1 > .001 or lo20 > .02) and edge > 0
    lines = [
        "NUMBERS3 過去回別ウォークフォワード精度", "=" * 92,
        f"対象回: 第{int(df.target_issue.iloc[0])}回 ～ 第{int(df.target_issue.iloc[-1])}回",
        f"対象期間: {df.draw_date.iloc[0]} ～ {df.draw_date.iloc[-1]}",
        f"評価回数: {n:,}回 / 初期学習: {warmup}回 / ML再学習間隔: {refit}回",
        "検証方式: 第N回は第N-1回までの情報だけを使用する prequential / walk-forward",
        "", "ストレート精度", "-" * 92,
    ]
    lines += [rate_line(k, m[f"h{k}"], n) for k in KS]
    lines += [
        "", "総合判定", "-" * 92,
        f"予測優位性: {'確認' if confirmed else '未確認'}",
        f"Top1一様0.1%は95%CI内={'YES' if lo1 <= .001 <= hi1 else 'NO'} / "
        f"Top20一様2.0%は95%CI内={'YES' if lo20 <= .02 <= hi20 else 'NO'} / "
        f"Exact NLL差(一様-model)={edge:+.8f}",
        "", "順位・確率損失", "-" * 92,
        f"実数字の平均順位: {m['mean_rank']:.2f}/1000 / 中央値: {m['median_rank']:.2f}/1000",
        f"Exact NLL平均: {m['exact_nll']:.8f} / 一様: {U_EXACT:.8f} / 差: {edge:+.8f}",
        f"1桁log loss: {m['digit_nll']:.8f} / 一様: {U_DIGIT:.8f}",
        f"Top20 BOX: {m['box20']}/{n} = {m['box20']/n:.4%}",
        f"Top20 ミニ: {m['mini20']}/{n} = {m['mini20']/n:.4%}",
        "", "独立した1口購入シナリオ（各回200円）", "-" * 92,
        "対象は過去データによる仮想検証。2方式は同時購入しません。",
        *[
            line for mode, name in (("straight", "ストレートのみ"), ("box", "ボックスのみ"))
            for line in (
                f"{name}: 的中={int(df[f'{mode}_ticket_hit'].sum())}/{n} ({df[f'{mode}_ticket_hit'].mean():.4%})",
                (f"  購入額={int(df[f'{mode}_ticket_stake'].sum()):,}円"
                 f" / 払戻={int(df[f'{mode}_ticket_return'].sum()):,}円"
                 f" / 損益={int(df[f'{mode}_ticket_profit'].sum()):+,}円"
                 f" / ROI={df[f'{mode}_ticket_profit'].sum()/df[f'{mode}_ticket_stake'].sum():+.4%}"
                 if df[f"{mode}_ticket_return"].notna().all()
                 else "  払戻データ欠損につき損益・ROI算出不可"),
            )
        ],
        "ボックスは3桁同一数字を除外し、次の購入可能な予測を採用。",
        "", "直近期間", "-" * 92,
    ]
    for w in (100, 500, 1000):
        x = metrics(df.tail(min(w, n)))
        lines.append(f"直近{x['n']:>4}回: Top1={x['r1']:.4%} Top5={x['r5']:.4%} "
                     f"Top20={x['r20']:.4%} ExactNLL={x['exact_nll']:.6f}")
    lines += ["", "年別精度", "-" * 92]
    years = pd.to_datetime(df.draw_date).dt.year
    for year in sorted(years.unique()):
        x = metrics(df[years == year])
        lines.append(f"{year}: n={x['n']:>3} Top1={x['r1']:.4%} Top5={x['r5']:.4%} "
                     f"Top20={x['r20']:.4%} ExactNLL={x['exact_nll']:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rounds(path, df):
    lines = ["NUMBERS3 過去回別予測一覧（未来情報リークなし）", "=" * 116,
             "回号 / 日付 / 実数字 / Straight1口 / Box1口 / S的中 / B的中 / 実数字順位 / S20 / B20 / M20 / Top20候補", ""]
    for _, r in df.sort_values("target_issue", ascending=False).iterrows():
        lines.append(f"第{int(r.target_issue):04d}回  {r.draw_date}  実={r.actual}  "
                     f"S={r.top1}  BOX={r.box_ticket_number} "
                     f"S1={'○' if r.straight_ticket_hit else '×'}  "
                     f"B1={'○' if r.box_ticket_hit else '×'}  "
                     f"順位={int(r.actual_rank):04d}/1000  "
                     f"S20={'○' if r.hit20 else '×'}  "
                     f"B20={'○' if r.box20 else '×'}  M20={'○' if r.mini20 else '×'}  候補={r.top20}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_historical_replay(csv_path: Path, outdir: Path, *, warmup_draws=500,
                          refit_interval=1000, top_k=20, ewma_alpha=.02,
                          force=False, random_state=42):
    if warmup_draws < 1 or refit_interval < 1 or top_k < 20:
        raise ValueError("warmup>=1, refit>=1, top_k>=20 が必要です")
    outdir.mkdir(parents=True, exist_ok=True)
    meta_p, idx_p = outdir / "historical_meta.json", outdir / "historical_index.json"
    summary_p, rounds_dir = outdir / "historical_accuracy.txt", outdir / "historical_rounds"
    df, nums = load_history(csv_path)
    fp = fingerprint(df)
    if not force and meta_p.exists() and idx_p.exists() and summary_p.exists() and rounds_dir.exists():
        try:
            old = json.loads(meta_p.read_text(encoding="utf-8"))
            idx = json.loads(idx_p.read_text(encoding="utf-8"))
            ok = bool(idx.get("chunks")) and all((outdir / c["text"]).exists() for c in idx["chunks"])
            if ok and old.get("dataset_fingerprint") == fp and old.get("warmup_draws") == warmup_draws \
                    and old.get("refit_interval") == refit_interval and old.get("top_k") == top_k:
                return {**old, "skipped": True}
        except Exception:
            pass

    X = build_features(nums)
    start = warmup_draws
    cfg = AgentConfig(top_k=top_k, random_state=random_state)
    losses = {k: U_DIGIT for k in MODEL_NAMES}
    rows = []

    # The first 99 predictions use only available past observations.
    # At 100 historical draws, fit ML for the next block; never fit on future data.
    segments = []
    if start < MIN_ML_TRAIN_DRAWS:
        segments.append((start, min(MIN_ML_TRAIN_DRAWS, len(df))))
    for bs in range(max(start, MIN_ML_TRAIN_DRAWS), len(df), refit_interval):
        segments.append((bs, min(bs + refit_interval, len(df))))
    for bs, be in segments:
        ml_ready = bs >= MIN_ML_TRAIN_DRAWS and all(
            np.unique(nums[5:bs, pos]).size >= 2 for pos in range(3)
        )
        if ml_ready:
            lr, et = fit_logreg(X, nums, bs, random_state), fit_trees(X, nums, bs, random_state)
            ml = predict_ml({"logreg": lr, "extra_trees": et}, X.iloc[bs:be])
        else:
            ml = None
        refit_issue = issue_no(df.iloc[bs - 1]["回別"])
        for j, t in enumerate(range(bs, be)):
            hist = nums[:t]
            freq = frequency_probs(hist)
            probs = {
                "frequency": freq[None, :, :],
                "markov": markov_probs(hist, freq)[None, :, :],
            }
            if ml is not None:
                probs.update({
                    "logreg": ml["logreg"][j:j+1],
                    "extra_trees": ml["extra_trees"][j:j+1],
                })
            raw_weights = adaptive_weights(losses, cfg.weight_temperature)
            weight_sum = sum(raw_weights[name] for name in probs)
            weights = {name: raw_weights[name] / weight_sum for name in probs}
            ens = ensemble_probs(probs, weights)[0]
            jp = joint_probs(ens)
            actual = num3(nums[t]); ai = int(actual); ap = float(jp[ai])
            rank = int(1 + np.sum(jp > ap))
            top = [n for n, _ in top_k_numbers(ens, top_k)]
            # Evaluate two separate bets, never add their stakes or payouts.
            box_choice = next((n for n in top if len(set(n)) > 1), None)
            if box_choice is None:
                raise RuntimeError("No eligible BOX number in the Top20")
            ticket_hit = int(actual == top[0])
            prize = pd.to_numeric(df.iloc[t].get("ストレート", np.nan), errors="coerce") if ticket_hit else 0.0
            ticket_return = float(prize) if pd.notna(prize) else np.nan
            box_hit = int(sorted(actual) == sorted(box_choice))
            box_prize = pd.to_numeric(df.iloc[t].get("ボックス", np.nan), errors="coerce") if box_hit else 0.0
            box_return = float(box_prize) if pd.notna(box_prize) else np.nan
            row = {
                "model_stage": "ensemble_ml" if ml is not None else "frequency_markov_only",
                "ticket_mode": "straight", "ticket_number": top[0],
                "ticket_count": 1, "ticket_stake": TICKET_PRICE_YEN,
                "ticket_hit": ticket_hit, "ticket_return": ticket_return,
                "ticket_profit": ticket_return - TICKET_PRICE_YEN if pd.notna(ticket_return) else np.nan,
                "straight_ticket_number": top[0],
                "straight_ticket_count": 1, "straight_ticket_stake": TICKET_PRICE_YEN,
                "straight_ticket_hit": ticket_hit, "straight_ticket_return": ticket_return,
                "straight_ticket_profit": ticket_return - TICKET_PRICE_YEN if pd.notna(ticket_return) else np.nan,
                "box_ticket_number": box_choice,
                "box_ticket_count": 1, "box_ticket_stake": TICKET_PRICE_YEN,
                "box_ticket_hit": box_hit, "box_ticket_return": box_return,
                "box_ticket_profit": box_return - TICKET_PRICE_YEN if pd.notna(box_return) else np.nan,
                "target_issue": issue_no(df.iloc[t]["回別"]),
                "draw_date": str(pd.to_datetime(df.iloc[t]["抽せん日"]).date()),
                "actual": actual, "refit_issue": refit_issue,
                "top1": top[0], "top20": " ".join(top[:20]), "actual_rank": rank,
                "exact_nll": -math.log(max(ap, 1e-15)), "digit_nll": digit_nll(nums[t], ens),
                "box20": int(any(sorted(x) == sorted(actual) for x in top[:20])),
                "mini20": int(any(x[-2:] == actual[-2:] for x in top[:20])),
            }
            for k in KS:
                row[f"hit{k}"] = int(actual in top[:k])
            rows.append(row)
            for name in probs:
                loss = digit_nll(nums[t], probs[name][0])
                losses[name] = (1 - ewma_alpha) * losses[name] + ewma_alpha * loss

    result = pd.DataFrame(rows)
    # Include all missed drawings in the expense and ROI denominator.
    result.to_csv(outdir / "historical_tickets.csv", index=False, encoding="utf-8-sig")
    write_summary(summary_p, result, warmup_draws, refit_interval)
    rounds_dir.mkdir(parents=True, exist_ok=True)
    for p in rounds_dir.glob("rounds_*.txt"):
        p.unlink()
    chunks = []
    for off in range(0, len(result), 1000):
        part = result.iloc[off:off+1000]
        a, b = int(part.target_issue.iloc[0]), int(part.target_issue.iloc[-1])
        rel = f"historical_rounds/rounds_{a:04d}_{b:04d}.txt"
        write_rounds(outdir / rel, part)
        chunks.append({"first_issue": a, "last_issue": b, "rows": len(part), "text": rel})
    idx_p.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "dataset_fingerprint": fp, "latest_issue": issue_no(df.iloc[-1]["回別"]),
        "evaluated_draws": len(result), "first_evaluated_issue": int(result.target_issue.iloc[0]),
        "last_evaluated_issue": int(result.target_issue.iloc[-1]), "warmup_draws": warmup_draws,
        "refit_interval": refit_interval, "top_k": top_k, "chunks": chunks, "skipped": False,
        "method": "past-only prequential walk-forward; frequency/Markov before 100 draws and periodic ML afterwards; causal adaptive weights",
        "purchase_policy": "independent_one_ticket_straight_vs_box",
        "ticket_count_per_scenario": 1, "unit_price_yen": TICKET_PRICE_YEN,
        "box_selection": "first ranked candidate with at least two distinct digits",
        "ticket_file": "historical_tickets.csv",
    }
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/numbers3.csv")
    ap.add_argument("--output-dir", default="backtests")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refit-interval", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run_historical_replay(Path(a.csv).resolve(), Path(a.output_dir).resolve(),
          warmup_draws=a.warmup, refit_interval=a.refit_interval, top_k=a.topk, force=a.force),
          ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
