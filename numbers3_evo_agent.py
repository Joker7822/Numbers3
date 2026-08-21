#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NUMBERS3 Evolutionary Prediction Agent

目的:
- numbers3.csv を読み込み
- 過去情報だけから特徴量を生成（未来リーク防止）
- 複数モデルを時系列ホールドアウトで評価
- 検証 log loss に応じてモデル重みを自動更新
- 全履歴で再学習して次回 000-999 の確率ランキングを出力
- state JSON に世代・重み・評価を保存

注意:
宝くじの抽せんは本質的に不確実です。本プログラムは当選を保証しません。
検証成績が一様分布ベースラインを上回らない場合、その事実も明示します。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

DIGITS = np.arange(10)
WINDOWS = (30, 100, 300, 1000)
LAGS = (1, 2, 3, 4, 5)
MODEL_NAMES = ("frequency", "markov", "logreg", "extra_trees")
EPS = 1e-12
UNIFORM_LOGLOSS = float(math.log(10.0))


@dataclass
class AgentConfig:
    validation_draws: int = 1000
    top_k: int = 20
    half_life_draws: int = 1500
    evolution_rate: float = 0.70
    weight_temperature: float = 0.02
    random_state: int = 42


def parse_number(value) -> Tuple[int, int, int]:
    ds = re.findall(r"\d", str(value))
    if len(ds) < 3:
        raise ValueError(f"本数字を3桁として解釈できません: {value!r}")
    return int(ds[0]), int(ds[1]), int(ds[2])


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=-1, keepdims=True)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = (
        str(len(df))
        + "|"
        + str(df.iloc[0]["抽せん日"])
        + "|"
        + str(df.iloc[-1]["抽せん日"])
        + "|"
        + str(df.iloc[-1]["本数字"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_history(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"抽せん日", "本数字", "回別"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSVに必須列がありません: {sorted(missing)}")

    df = df.copy()
    df["_date"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    if df["_date"].isna().any():
        bad = int(df["_date"].isna().sum())
        raise ValueError(f"抽せん日に解釈不能値があります: {bad}件")

    df = df.sort_values(["_date", "回別"]).drop_duplicates("抽せん日", keep="last")
    nums = np.array([parse_number(x) for x in df["本数字"]], dtype=int)

    if len(df) < 500:
        raise ValueError("学習には最低500回分を推奨します。")
    return df.reset_index(drop=True), nums


def preserve_merge_scraped(csv_path: Path, scraper_path: Path) -> int:
    """
    既存 scraper の取得関数だけを利用し、CSV保存はこのエージェント側で実施。
    既存CSVの追加列を落とさない。
    """
    spec = importlib.util.spec_from_file_location("numbers3_scraper_external", scraper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"scraperをロードできません: {scraper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    driver = mod.build_driver(headless=True)
    try:
        driver.get(mod.URL)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        import time
        time.sleep(0.8)
        mod.wait_for_tables(driver, timeout_total=45, retries=2)
        rows = mod.parse_tables(driver)
    finally:
        driver.quit()

    if not rows:
        return 0

    old = pd.read_csv(csv_path)
    existing_dates = set(old["抽せん日"].astype(str))
    new_rows = [r for r in rows if str(r.get("抽せん日")) not in existing_dates]
    if not new_rows:
        return 0

    all_cols = list(old.columns)
    for r in new_rows:
        for c in r:
            if c not in all_cols:
                all_cols.append(c)

    expanded = []
    for r in new_rows:
        x = {c: np.nan for c in all_cols}
        x.update(r)
        if "払戻金完全性" in all_cols:
            prize_cols = ["ストレート", "ボックス", "セット(ストレート)", "セット(ボックス)", "ミニ"]
            x["払戻金完全性"] = int(all(pd.notna(x.get(c)) for c in prize_cols))
        if "データソース" in all_cols:
            x["データソース"] = "agent_scrape"
        expanded.append(x)

    merged = pd.concat([old.reindex(columns=all_cols), pd.DataFrame(expanded, columns=all_cols)], ignore_index=True)
    merged["_date"] = pd.to_datetime(merged["抽せん日"], errors="coerce")
    merged = merged.sort_values(["_date", "回別"]).drop(columns=["_date"])
    merged = merged.drop_duplicates("抽せん日", keep="last")
    backup = csv_path.with_suffix(csv_path.suffix + ".bak")
    old.to_csv(backup, index=False, encoding="utf-8")
    merged.to_csv(csv_path, index=False, encoding="utf-8")
    return len(new_rows)


def build_features(nums: np.ndarray) -> pd.DataFrame:
    n = len(nums)
    idx = pd.RangeIndex(n)
    feat = {}

    for pos in range(3):
        s = pd.Series(nums[:, pos], index=idx)
        for lag in LAGS:
            shifted = s.shift(lag)
            for d in DIGITS:
                feat[f"p{pos}_lag{lag}_is{d}"] = (shifted == d).astype(float)

    for pos in range(3):
        s = pd.Series(nums[:, pos], index=idx)
        for w in WINDOWS:
            for d in DIGITS:
                ind = (s == d).astype(float).shift(1)
                feat[f"p{pos}_freq_w{w}_d{d}"] = ind.rolling(w, min_periods=5).mean()

    for d in DIGITS:
        per_draw = pd.Series((nums == d).sum(axis=1) / 3.0, index=idx)
        for w in WINDOWS:
            feat[f"all_freq_w{w}_d{d}"] = per_draw.shift(1).rolling(w, min_periods=5).mean()

    sums = nums.sum(axis=1) / 27.0
    unique = np.array([len(set(row)) for row in nums], dtype=float) / 3.0
    odd = (nums % 2).sum(axis=1) / 3.0
    spread = (nums.max(axis=1) - nums.min(axis=1)) / 9.0
    adjacent_same = ((nums[:, 0] == nums[:, 1]) | (nums[:, 1] == nums[:, 2])).astype(float)
    for name, arr in [
        ("prev_sum", sums),
        ("prev_unique", unique),
        ("prev_odd", odd),
        ("prev_spread", spread),
        ("prev_adj_same", adjacent_same),
    ]:
        feat[name] = pd.Series(arr, index=idx).shift(1)

    X = pd.DataFrame(feat, index=idx)
    X = X.fillna(0.1)
    return X


def make_next_features(nums: np.ndarray) -> pd.DataFrame:
    extended = np.vstack([nums, np.array([[-1, -1, -1]], dtype=int)])
    X = build_features(extended)
    return X.iloc[[-1]].copy()


def dirichlet_freq(history: np.ndarray, pos: int, window: int, alpha: float = 1.0) -> np.ndarray:
    h = history[-window:, pos] if len(history) > window else history[:, pos]
    counts = np.bincount(h, minlength=10).astype(float)
    return normalize_probs(counts + alpha)


def frequency_probs(history: np.ndarray) -> np.ndarray:
    ws = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
    out = np.zeros((3, 10), dtype=float)
    for pos in range(3):
        for a, w in zip(ws, WINDOWS):
            out[pos] += a * dirichlet_freq(history, pos, w)
    return normalize_probs(out)


def markov_probs(history: np.ndarray, trans_window: int = 1000) -> np.ndarray:
    base = frequency_probs(history)
    out = np.zeros((3, 10), dtype=float)
    if len(history) < 3:
        return base
    h = history[-(trans_window + 1):] if len(history) > trans_window + 1 else history
    for pos in range(3):
        prev = int(history[-1, pos])
        from_d = h[:-1, pos]
        to_d = h[1:, pos]
        counts = np.bincount(to_d[from_d == prev], minlength=10).astype(float)
        conditional = normalize_probs(counts + 1.0)
        out[pos] = 0.65 * conditional + 0.35 * base[pos]
    return normalize_probs(out)


def heuristic_validation_probs(nums: np.ndarray, indices: np.ndarray) -> Dict[str, np.ndarray]:
    freq = np.zeros((len(indices), 3, 10), dtype=float)
    markov = np.zeros_like(freq)
    for j, t in enumerate(indices):
        hist = nums[:t]
        freq[j] = frequency_probs(hist)
        markov[j] = markov_probs(hist)
    return {"frequency": freq, "markov": markov}


def recency_sample_weight(n: int, half_life: int) -> np.ndarray:
    age = (n - 1) - np.arange(n)
    return np.power(0.5, age / float(half_life))


def fit_ml_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg: AgentConfig,
) -> Dict[str, List]:
    sw = recency_sample_weight(len(X_train), cfg.half_life_draws)
    lr_models, et_models = [], []
    for pos in range(3):
        lr = LogisticRegression(
            C=0.25,
            max_iter=700,
            solver="lbfgs",
            random_state=cfg.random_state,
        )
        lr.fit(X_train, y_train[:, pos], sample_weight=sw)
        lr_models.append(lr)

        et = ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=cfg.random_state + pos,
            class_weight=None,
        )
        et.fit(X_train, y_train[:, pos], sample_weight=sw)
        et_models.append(et)
    return {"logreg": lr_models, "extra_trees": et_models}


def predict_ml(models: Dict[str, List], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    result = {}
    for name, per_pos in models.items():
        arr = np.zeros((len(X), 3, 10), dtype=float)
        for pos, model in enumerate(per_pos):
            p = model.predict_proba(X)
            for col, klass in enumerate(model.classes_):
                arr[:, pos, int(klass)] = p[:, col]
            arr[:, pos, :] = normalize_probs(arr[:, pos, :])
        result[name] = arr
    return result


def mean_position_logloss(y: np.ndarray, probs: np.ndarray) -> float:
    losses = []
    for pos in range(3):
        losses.append(log_loss(y[:, pos], probs[:, pos, :], labels=list(range(10))))
    return float(np.mean(losses))


def top_k_numbers(prob3: np.ndarray, k: int) -> List[Tuple[str, float]]:
    joint = (
        prob3[0, :, None, None]
        * prob3[1, None, :, None]
        * prob3[2, None, None, :]
    ).reshape(-1)
    k = min(k, 1000)
    idx = np.argpartition(joint, -k)[-k:]
    idx = idx[np.argsort(joint[idx])[::-1]]
    out = []
    for z in idx:
        number = f"{int(z):03d}"
        out.append((number, float(joint[z])))
    return out


def topk_hit_rate(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    hits = 0
    for i in range(len(y)):
        actual = "".join(map(str, y[i]))
        top = {n for n, _ in top_k_numbers(probs[i], k)}
        hits += actual in top
    return hits / len(y)


def derive_weights(losses: Dict[str, float], temperature: float) -> Dict[str, float]:
    best = min(losses.values())
    raw = {
        name: math.exp(-(loss - best) / max(temperature, 1e-6))
        for name, loss in losses.items()
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def blend_weights(
    current: Dict[str, float],
    previous: Dict[str, float] | None,
    evolution_rate: float,
) -> Dict[str, float]:
    if not previous:
        return current
    keys = sorted(set(current) | set(previous))
    blended = {
        k: evolution_rate * current.get(k, 0.0)
        + (1.0 - evolution_rate) * previous.get(k, 0.0)
        for k in keys
    }
    s = sum(blended.values())
    return {k: v / s for k, v in blended.items()}


def ensemble_probs(all_probs: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    names = [n for n in MODEL_NAMES if n in all_probs and n in weights]
    out = np.zeros_like(all_probs[names[0]], dtype=float)
    for name in names:
        out += weights[name] * all_probs[name]
    denom = out.sum(axis=-1, keepdims=True)
    return np.clip(out, EPS, None) / np.clip(denom, EPS, None)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def next_weekday(date_value: pd.Timestamp) -> pd.Timestamp:
    d = date_value + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def run_agent(csv_path: Path, outdir: Path, state_path: Path, cfg: AgentConfig) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    df, nums = load_history(csv_path)
    fp = dataset_fingerprint(df)

    X = build_features(nums)
    start = max(LAGS)
    validation_draws = min(cfg.validation_draws, max(200, len(df) // 4))
    split = len(df) - validation_draws
    if split <= start + 200:
        raise ValueError("時系列検証用の学習データが不足しています。")

    train_idx = np.arange(start, split)
    val_idx = np.arange(split, len(df))
    X_train = X.iloc[train_idx]
    y_train = nums[train_idx]
    X_val = X.iloc[val_idx]
    y_val = nums[val_idx]

    ml = fit_ml_models(X_train, y_train, cfg)
    val_probs = heuristic_validation_probs(nums, val_idx)
    val_probs.update(predict_ml(ml, X_val))

    losses = {name: mean_position_logloss(y_val, p) for name, p in val_probs.items()}
    current_weights = derive_weights(losses, cfg.weight_temperature)

    prev_state = load_state(state_path)
    same_data = prev_state.get("dataset_fingerprint") == fp
    prev_weights = prev_state.get("model_weights") if (prev_state and not same_data) else None
    weights = blend_weights(current_weights, prev_weights, cfg.evolution_rate)

    ens_val = ensemble_probs(val_probs, weights)
    ens_loss = mean_position_logloss(y_val, ens_val)
    top1_rate = topk_hit_rate(y_val, ens_val, 1)
    top20_rate = topk_hit_rate(y_val, ens_val, 20)

    full_idx = np.arange(start, len(df))
    full_models = fit_ml_models(X.iloc[full_idx], nums[full_idx], cfg)
    X_next = make_next_features(nums)

    next_probs = {
        "frequency": frequency_probs(nums)[None, :, :],
        "markov": markov_probs(nums)[None, :, :],
    }
    next_probs.update(predict_ml(full_models, X_next))
    ens_next = ensemble_probs(next_probs, weights)[0]

    ranking = top_k_numbers(ens_next, cfg.top_k)
    rank_df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "number": n,
                "model_probability": p,
                "uniform_probability": 1.0 / 1000.0,
                "relative_to_uniform": p / 0.001,
                "d1_probability": ens_next[0, int(n[0])],
                "d2_probability": ens_next[1, int(n[1])],
                "d3_probability": ens_next[2, int(n[2])],
            }
            for i, (n, p) in enumerate(ranking)
        ]
    )
    ranking_path = outdir / "numbers3_prediction_latest.csv"
    rank_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    last_date = pd.to_datetime(df.iloc[-1]["抽せん日"])
    model_next_date = next_weekday(last_date)
    edge = UNIFORM_LOGLOSS - ens_loss

    generation = int(prev_state.get("generation", 0))
    if not same_data:
        generation += 1
    generation = max(generation, 1)

    report = {
        "generated_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "generation": generation,
        "dataset_fingerprint": fp,
        "dataset": {
            "rows": int(len(df)),
            "first_draw_date": str(pd.to_datetime(df.iloc[0]["抽せん日"]).date()),
            "last_draw_date": str(last_date.date()),
            "last_issue": str(df.iloc[-1]["回別"]),
            "last_number": "".join(map(str, nums[-1])),
            "model_next_weekday_candidate": str(model_next_date.date()),
        },
        "validation": {
            "draws": int(len(val_idx)),
            "period_start": str(pd.to_datetime(df.iloc[split]["抽せん日"]).date()),
            "period_end": str(last_date.date()),
            "model_logloss": losses,
            "ensemble_logloss": ens_loss,
            "uniform_logloss": UNIFORM_LOGLOSS,
            "logloss_improvement_vs_uniform": edge,
            "top1_straight_hit_rate": top1_rate,
            "top20_straight_hit_rate": top20_rate,
            "uniform_expected_top1_rate": 0.001,
            "uniform_expected_top20_rate": 0.02,
            "edge_status": "positive" if edge > 0 else "not_confirmed",
        },
        "model_weights": weights,
        "current_validation_weights": current_weights,
        "top_predictions": [
            {"rank": i + 1, "number": n, "probability": p}
            for i, (n, p) in enumerate(ranking)
        ],
        "notes": [
            "未来情報を特徴量に含めない時系列ホールドアウト評価。",
            "モデル重みは検証log lossから自動算出し、新規データ世代では前世代重みと平滑化。",
            "検証成績が一様分布を上回らない場合、予測優位性は確認できない。",
            "model_next_weekday_candidate は単純な平日計算で、祝日・年末年始は未反映。",
        ],
    }

    report_path = outdir / "numbers3_report_latest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    state = {
        "generation": generation,
        "dataset_fingerprint": fp,
        "last_draw_date": str(last_date.date()),
        "model_weights": weights,
        "last_validation": report["validation"],
        "last_top_predictions": report["top_predictions"],
        "updated_at": report["generated_at"],
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main():
    ap = argparse.ArgumentParser(description="NUMBERS3 自己適応型予測エージェント")
    ap.add_argument("--csv", default="numbers3.csv", help="履歴CSV")
    ap.add_argument("--outdir", default="numbers3_agent_output", help="出力ディレクトリ")
    ap.add_argument("--state", default="numbers3_agent_state.json", help="進化状態JSON")
    ap.add_argument("--validation", type=int, default=1000, help="時系列検証に使う直近回数")
    ap.add_argument("--topk", type=int, default=20, help="出力候補数")
    ap.add_argument("--refresh", action="store_true", help="外部scraperを使ってCSVを先に更新")
    ap.add_argument("--scraper", default="scrapingnumbers3.py", help="既存scraperのパス")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    outdir = Path(args.outdir).resolve()
    state_path = Path(args.state).resolve()

    if args.refresh:
        added = preserve_merge_scraped(csv_path, Path(args.scraper).resolve())
        print(f"[REFRESH] 新規 {added} 件。既存追加列は保持しました。")

    cfg = AgentConfig(validation_draws=args.validation, top_k=args.topk)
    report = run_agent(csv_path, outdir, state_path, cfg)

    print(json.dumps({
        "generation": report["generation"],
        "dataset": report["dataset"],
        "validation": report["validation"],
        "model_weights": report["model_weights"],
        "top_predictions": report["top_predictions"][:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
