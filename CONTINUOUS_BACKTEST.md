# NUMBERS3 連続バックテスト

バックテスト完了後、待機せず直ちに次のバックテストを開始する常時ループです。

## 動作

- Workflow: `.github/workflows/numbers3-continuous-backtest.yml`
- 1サイクル: `numbers3_continuous_backtest.py` → `numbers3_historical_backtest.py --force`
- 検証方式: 第N回の予測には第N-1回までのデータだけを使う walk-forward / prequential
- サイクル間の sleep: **なし**
- 同時実行数: **1** (`concurrency`)
- 最新の履歴・コードの取り込み: 30分ごとのチェックポイント時
- チェックポイント: 30分ごと + workflow終了前
- GitHub-hosted runnerの6時間上限より前に、5時間30分でループを終了
- 終了後 `workflow_dispatch` で次のrunを直ちに起動
- 6時間ごとのcronをwatchdogとして設定し、失敗等で連鎖が止まった場合に再始動

## 永続ファイル

- `backtests/continuous_status.txt`
  - 累計サイクル、最新評価回、Top1/Top20、Exact NLL、所要時間
- `backtests/continuous_backtest_history.csv`
  - 連続バックテスト履歴。リポジトリ肥大化防止のため直近10,000サイクルを保持
- `state/continuous_backtest_state.json`
  - 累計サイクル数などを永続保存

## 権限

継続運転に必要な範囲だけを付与します。

```yaml
permissions:
  contents: write
  actions: write
```

- `contents: write`: チェックポイント結果をmainへ保存
- `actions: write`: 終了直前に次の `workflow_dispatch` runを起動

## 注意

同一データ・同一コードを繰り返しバックテストした結果は、互いに独立した統計標本ではありません。連続実行は再現性・稼働監視・新しい抽せんデータ/コード変更の即時再評価に使います。繰り返した回数だけ統計的な予測精度が高くなったとは扱いません。
