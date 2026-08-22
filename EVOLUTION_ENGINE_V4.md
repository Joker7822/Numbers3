# NUMBERS3 Evolution Engine v4

v4はv3のrolling 5-fold、paired bootstrap、confirmation seed、final audit隔離、live-only評価を維持しつつ、確率校正・Champion平均・探索効率を改善します。

## 1. Power calibration

`uniform_blend` の後に `power_beta` を適用します。

`P'(x) = P(x)^beta / sum(P(j)^beta)`

- beta < 1: 分布を平坦化し、過信を抑える
- beta = 1: 変更なし
- beta > 1: 上位確率を強調

`power_beta` はChallenger探索対象で、final auditを見て選択しません。

## 2. Champion Pool

Primary Championの昇格ルールは単体比較のまま維持します。
実際の次回予測は、昇格済みChampionを最大5モデル保持するChampion Poolから生成します。

Pool重みはdevelopment Exact NLLだけから算出し、final auditやlive結果は重み決定に使いません。
全履歴リプレイも同じPoolで実行するため、本番予測と履歴評価のモデル定義を一致させます。

## 3. Meta-model guided search

過去の `experiments/experiment_history.csv` を教師データとして、Challenger設定からdevelopment NLLを予測するExtraTreesRegressorを学習します。

各実験で192個の未試行候補を生成し、通常はLower Confidence Boundで候補を選択します。
10回に1回は予測不確実性が最大の候補を選び、探索を維持します。
Meta-modelの予測は候補選択だけに使い、Champion昇格は必ず本物のrolling 5-foldで判定します。

## 4. 多重探索対策

Adaptive trial数と世代に応じてpaired bootstrapの必要閾値を厳しくします。
これはdevelopmentデータ再利用による選択バイアスを完全に消すものではないため、live-only leaderboardを最終的な外部評価として維持します。

## 5. 目標

第一目標は causal selection Exact NLL を一様基準 `ln(1000) = 6.907755...` より小さくすることです。
ただし過去データ上で基準を下回っても、未知の抽せんに対する優位性を保証しません。
