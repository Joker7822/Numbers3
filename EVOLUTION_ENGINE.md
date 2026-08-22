# NUMBERS3 Evolution Engine v2

## 目的

連続バックテストを、同一条件の再実行ではなく **Challenger生成 → Nested Walk-Forward → Champion比較 → 昇格/却下** の連続実験に変更します。

## 1サイクル

1. 現在の `state/evolution_champion.json` を読み込む
2. 未実験のChallenger設定を自動生成する
3. 最終1,000回を final audit holdout として隔離する
4. その直前1,500回を500回×3 foldに分ける
5. 各fold内の前半だけでアンサンブル重みを決め、後半だけを採点する
6. Primary seedでChampionより改善した候補だけconfirmation seedでも再評価する
7. 2 seedで改善が再現し、3 fold中2 fold以上で勝ち、最大悪化幅が閾値内なら昇格する
8. Final auditは報告だけに使い、昇格判定には絶対に使わない
9. 実験結果を `experiments/experiment_history.csv` に保存する
10. 次のChallengerを待機なしで直ちに生成する

## 探索対象

- 学習データの半減期
- アンサンブル進化率
- モデル重みtemperature
- 頻度モデルの短期/中期/長期比率
- Markov混合比
- Markov遷移窓
- Logistic Regressionの正則化
- ExtraTreesの木数
- ExtraTreesの最小leaf

## Champion / Challenger

本番の日次予測は `numbers3_runner_champion.py` を通じて、現在のChampion設定を `numbers3_evo_agent.py` に注入します。

Challengerはdevelopment foldで以下を満たした場合のみ昇格候補になります。

- Exact NLL改善が閾値以上
- 3 fold中2 fold以上でChampionより良い
- 1 foldだけ極端に悪化していない
- 別random seedでも改善が再現する

Final audit holdoutの結果は表示しますが、昇格判定には使用しません。これによりholdoutを見ながらモデルを選ぶ過学習を防ぎます。

## 出力

- `state/evolution_champion.json` — 現在のChampion設定と評価キャッシュ
- `state/evolution_engine_state.json` — 累計実験数と現在世代
- `experiments/experiment_history.csv` — 全Challengerの設定・評価・採否
- `experiments/latest_experiment.txt` — 最新実験の人間可読結果
- `backtests/continuous_status.txt` — 既存監視先との互換表示

## 連続運転

`.github/workflows/numbers3-continuous-backtest.yml` は約5時間30分の間、sleepなしで実験を繰り返します。GitHub-hosted runnerの上限前に自分自身を `workflow_dispatch` し、次runへ引き継ぎます。6時間ごとのscheduleは停止時のwatchdogです。

同じデータでも各サイクルで異なるChallengerを評価するため、計算資源は再現性確認だけでなくモデル探索に使われます。

## 重要

改善判定は「当たった回数が増えた」だけでは行いません。主目的はout-of-sample Exact NLLで、Top1/Top20は補助指標です。宝くじの抽せんに持続的な予測可能性が存在する保証はありません。改善が再現しないChallengerは自動的に却下されます。
