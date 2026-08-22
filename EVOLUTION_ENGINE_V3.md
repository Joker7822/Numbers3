# NUMBERS3 Evolution Engine v3

## 目的

v2で5,000回以上のChallenger探索が進んでもChampion更新が停滞したため、計算回数を増やすのではなく、探索空間・評価・過学習対策を更新する。

## 主な変更

### 1. 実験停滞を直接カウント

`source_experiment_id` から直近Champion昇格実験を特定し、`experiments_since_last_promotion` を計算する。リプレイがcache skipされても停滞が増えるため、旧`stagnation_count`依存の問題を避ける。

探索モード:

- 0–9: local（1–2パラメータ）
- 10–49: wider（2–3）
- 50–199: broad（3–5）
- 200–999: global（4–7）
- 1000以上: structural（5–9、構造パラメータを最低1個含む）

### 2. Uniform shrinkage

最終joint確率を `P_final = λ * P_model + (1-λ) * Uniform(000..999)` で縮小する。`uniform_blend=λ` はChallenger探索対象。過信を抑え、Exact NLLを主目的として校正する。

### 3. Rolling 5-fold

最終1,000回をfinal auditとして隔離し、その直前1,500回を300回×5 foldで評価。各foldは前半でensemble weightを校正、後半だけ採点する。各fold開始時点より前のデータだけでMLをfitする。

### 4. Paired bootstrap昇格判定

同じ対象回についてChampionとChallengerのExact NLL差をpaired bootstrapし、改善が再現される確率を評価。adaptive trialが増えた場合は必要確率を最大99%まで強化する。

昇格には、Exact NLL改善量、5 fold中4 fold以上勝利、最大fold悪化幅、paired bootstrap、confirmation seedの条件をすべて要求する。final auditは報告専用で昇格には使わない。

### 5. 3桁Autoregressive component

桁独立jointに加え、過去履歴から `P(百) × P(十|百) × P(一|百,十)` をDirichlet平滑化・recency weightingで構築する。`autoregressive_mix` と `ar_alpha` を探索する。

### 6. Feature evolution

`feature_profile` をChampion遺伝子に追加する。

- `base`
- `compact`
- `gap`
- `gap_pair`
- `gap_pair_shape`

未出現間隔、前回2桁pair、rolling shape統計を必要に応じて追加し、特徴量構造そのものをChallenger探索する。

### 7. Live-only leaderboard

`predictions/live_forecast_archive.csv` に、抽せん前に生成した1000通りの確率分布をChampion ID付きで保存する。抽せん後、`predictions/live_scoreboard.csv` / `.txt` で Exact NLL、実数字順位、Top1/Top20、Champion generation別成績を集計する。

バックテスト対象として既に見た回ではなく、実際に予測を保存した後に抽せんされた回だけを採点する。抽せん当日生成の予測は厳格に除外する。

## 運用

Continuous Evolutionはv3 engineを連続実行し、Champion昇格時にv3 full-history replayを直ちに実行する。その後Daily Predictionをdispatchし、新Championで次回予測を生成する。

## 注意

NUMBERS3は本質的にランダムな抽せんであり、探索・バックテストの改善が未知回での予測優位性を保証するものではない。特に多数のadaptive trialは過学習を生み得るため、final auditとlive-only成績を分離して扱う。
