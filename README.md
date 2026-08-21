# Numbers3 — 自己適応型予測AIエージェント

NUMBERS3 の過去抽せんデータを継続更新し、時系列バックテストでモデルを評価しながら
次回候補を自動生成するリポジトリです。

## 自動実行

GitHub Actions は **日本時間の日曜日〜木曜日、毎日20:00** に実行されます。

- GitHub Actions cron: `0 11 * * 0-4`
- GitHub cron は UTC のため、11:00 UTC = 20:00 JST
- `workflow_dispatch` でも手動実行可能
- Actions は `contents: write` 権限で、更新データ・予測・バックテスト・状態をリポジトリへコミットします

## 1回の自動実行で行うこと

1. みずほ銀行 NUMBERS3 ページから最新抽せんを取得
2. `data/numbers3.csv.gz` を展開して履歴を更新し、実行後に再圧縮
3. 過去の未照合予測を **回号** で当選結果と照合
4. 直近1,000回をホールドアウトした時系列バックテストを再実行
5. frequency / Markov / Logistic Regression / ExtraTrees を評価
6. 検証 log loss に応じてアンサンブル重みを更新
7. 全履歴で再学習
8. 次回の上位20候補を生成
9. 累積予測・照合結果・バックテスト履歴を保存
10. 生成物を自動コミット

## 閲覧するファイル

### 最新予測

[`predictions/latest_prediction.txt`](predictions/latest_prediction.txt)

人がそのまま読める最新予測、バックテスト、モデル重み、上位候補です。

### 累積予測と当選照合

[`predictions/cumulative_results.txt`](predictions/cumulative_results.txt)

過去に出した各予測を、実際の本数字と照合したテキストです。
当選金額は「その購入方式で買っていた場合」の方式別払戻金として表示します。

### 全予測の機械可読履歴

[`predictions/prediction_history.csv`](predictions/prediction_history.csv)

各回・各順位の予測確率、照合状態、本数字、成立した購入方式、払戻金を累積保存します。

### バックテスト履歴

[`backtests/backtest_history.csv`](backtests/backtest_history.csv)

世代ごとの log loss、一様分布との差、Top1/Top20包含率、モデル重みを保存します。

## 当選照合の考え方

予測数字に対して以下を別々に判定します。

- **ストレート**: 3桁・並び順とも完全一致
- **ボックス**: 3桁の数字集合が一致
- **セット(ストレート)**: 完全一致時のセット・ストレート払戻
- **セット(ボックス)**: 並べ替え一致時のセット・ボックス払戻
- **ミニ**: 下2桁が一致

1つの予測数字でも購入方式によって払戻金が異なるため、
単一の「当選金額」とはせず、成立する方式ごとの金額を表示します。

## モデルの自己適応

このエージェントの「自己進化」は、コードを無制御に自己改変する方式ではありません。

- 新しい抽せんデータを取り込む
- 未学習期間で各モデルを測定する
- log loss が良いモデルの重みを上げる
- 悪いモデルの重みを下げる
- 前世代の重みと平滑化する
- 再学習して次回予測する

これにより、変更の良し悪しをバックテスト数値で追跡できます。

## 権限

Actions の実行トークンには **生成物をリポジトリへ保存するための `contents: write`** を付与しています。
リポジトリ設定・Secrets・Issues・Packages・Deployments 等への無関係な `write-all` は使用していません。
これはAIエージェントが本タスクを完全自動化するために必要な書込権限を持ちつつ、
漏えい時の影響範囲を最小化するためです。

## 手動実行

```bash
pip install -r requirements.txt

python numbers3_runner.py \
  --csv data/numbers3.csv \
  --scraper scrapingnumbers3.py \
  --refresh \
  --validation 1000 \
  --topk 20
```

## 注意

宝くじの当選を保証するものではありません。抽せんが独立・一様である場合、
過去履歴から持続的な予測優位性を得られない可能性があります。
エージェントは一様分布を基準に継続バックテストし、優位性が確認できない場合もそのまま記録します。


## 履歴データ

GitHub上では `data/numbers3.csv.gz` として圧縮保存し、Actions実行時だけ `data/numbers3.csv` に展開します。これにより履歴全体を保持しつつ、コミット容量を抑えます。
