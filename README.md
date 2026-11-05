# Numbers3 — 自己適応型予測AIエージェント

NUMBERS3 の過去抽せんデータを継続更新し、時系列バックテストでモデルを評価しながら次回候補を自動生成するリポジトリです。

## BOXの的中率強化・比較実験（2026-11-05）

従来のBOXは「個々の順序付き3桁の予測順位が最上位の購入可能数字」を1口選びます。ただしBOXの的中条件は順列全体です。将来の候補として二つのシャドー方式を検証します。

- **順列確率合計最大**：3桁の順列全体（異なる数字なら6通り・同じ数字2個なら3通り）のモデル確率を合計し、最も高いBOXグループを選びます。
- **3桁すべて異なる・確率合計最大**：6通りのBOXグループだけを候補にして、予測確率合計が最大のグループを選びます。理論的に均等な抽せんでの的中確率は0.6%です。2桁同一のBOXは0.3%です。

選び方の評価には当選番号も払戻金も使わず、予測時点以前の情報だけで生成された確率を使います。全履歴のwalk-forwardと、過去に販売締切前に保存したLive確率への**事後設計した選び方の再適用**を、既存方式とは独立に集計します。これは実際に当時この政策を使って予測していたLive運用記録ではありません。比較結果は後から考えた方式の探索を含むため、有意性が出ても独立した将来検証が必要です。**正式な購入方式と過去の収支集計は変更しません。**

**初回比較結果（2026-11-05、対象第2回〜第7107回）**: BOX従来45/7106、確率合計最大48/7106、3桁相異なる方式48/7106。後二者の差は+3回ですが対応のある探索的検定p=0.8243で有意な改善ではありません。第5000回以降は従来15/2108、両候補11/2108で悪化。保存済み抽せん前確率51回への後付け比較は従来1回、両候補1回。**正式採用は見送り**です。今後の抽せん前に政策を確定し、十分な期間で独立検証することが必要です。

実装: `numbers3_hit_rate_lab.py`、テスト: `test_numbers3_hit_rate_lab.py`、結果: `dashboard/data/summary.json` の `box_shadows`。予測側は将来の「検証中BOX候補」を別枠で出し、既存のストレート／BOX実績とは混ぜません。

出典: [みずほ銀行 ナンバーズのルール](https://www.mizuhobank.co.jp/tenpoinfo/atm/takarakuji/t02.html)。

## 可視化ダッシュボード（1口ストレート vs 1口ボックス）

[`dashboard/index.html`](dashboard/index.html) にスマートフォン対応の検証画面を用意しています。
第2回からの過去回別ウォークフォワードと、実際に抽せん前に保存したLive予測を**混ぜずに**切り替えます。
直近1か月の当選数字・予測照合、桁別出現回数、次回予測も表示します。

- **ストレートのみ**: 各回の予測1位を200円・1口購入したと仮定し、3桁完全一致で的中
- **ボックスのみ**: 各回の購入可能な予測の最上位を200円・1口購入したと仮定し、順序を問わず3桁一致で的中。全桁同一（例111）はボックス購入できないため次候補へ
- 両方式の**購入額・払戻金・利益・ROIは独立**。2方式を同時購入した費用や収支は算出しない
- 払戻金は各抽せん回の「ストレート」「ボックス」列を使用する。古いデータで的中時の払戻金が未記録の回は、日付・回号・当選数字を照合した `data/numbers3_historical_payouts.csv`（くじしるべ公開アーカイブの二次資料）で復元する。45回分を収録するが、全件の公式独立照合は未完了。なお未補完の的中があれば利益・ROIは「算出不可」とする
- 第2回からの復元では第1回だけを観測済みとして開始。初期期間は頻度・マルコフに限定し、100回以上蓄積してから過去データで機械学習を学習。現在のChampionモデルを第2回当時から使ったとみなす検証ではない

データを自動生成するワークフロー: [`numbers3-single-ticket-backtest.yml`](.github/workflows/numbers3-single-ticket-backtest.yml)

出力先:
- `backtests/one_ticket_from_second/historical_accuracy.txt`（第2回からの方式別総計）
- `backtests/one_ticket_from_second/historical_tickets.csv`（各回の購入数字・的中・払戻・損益）
- `data/numbers3_historical_payouts.csv`（過去45回の復元金額と個別出典URL。抽せん日前の予測特徴量には利用しない）
- `predictions/live_purchase_scoreboard.csv`（実保存予測の方式別Live照合）
- `dashboard/data/summary.json`（ダッシュボードの表示データ）

ローカルで閲覧する場合は、リポジトリのルートで `python -m http.server 8000` を実行し、
`http://localhost:8000/dashboard/` にアクセスします。GitHub Pages を使う場合は
`dashboard/` 以下を配信対象にしてください。初回は単口監査ワークフローが完了してから数値が生成されます。

手元で全回の払戻金を再照合する場合は、バックテスト実行後に `python numbers3_payout_reconciliation.py` を実行し、次に `python numbers3_dashboard_data.py` で表示用JSONを再生成します。復元金額は [くじしるべ](https://kujishirube.info/numbers3/csv/) の公表値であり、[楽天宝くじの公開履歴](https://takarakuji.rakuten.co.jp/backnumber/numbers3/202402/) では一部の回を照合しました。

**注意:** 過去再検証は当時購入した記録ではありません。また、方式ごとの単発的な好成績だけでは統計的な予測優位性を示せません。

## 自動実行

GitHub Actions は **日本時間の日曜日〜木曜日、毎日20:00** に実行されます。

- GitHub Actions cron: `0 11 * * 0-4`
- GitHub cron は UTC のため、11:00 UTC = 20:00 JST
- `workflow_dispatch` でも手動実行可能
- Actions は `contents: write` 権限で、更新データ・予測・バックテスト・状態をリポジトリへコミットします

## 1回の自動実行で行うこと

1. `data/history_parts/part_*` から全履歴CSVを復元
2. みずほ銀行 NUMBERS3 ページから最新抽せんを取得
3. 過去の未照合予測を **回号** で当選結果と照合
4. **過去回別 walk-forward 監査**を実行（履歴が変わった時だけ再計算）
5. 直近1,000回をホールドアウトした時系列バックテストを再実行
6. frequency / Markov / Logistic Regression / ExtraTrees を評価
7. 検証 log loss に応じてアンサンブル重みを更新
8. 全履歴で再学習
9. 次回の上位20候補を生成
10. 累積予測・照合結果・バックテスト履歴を保存
11. 更新後の履歴を XZ 圧縮 + Base64 化して再分割
12. 生成物を自動コミット

## 閲覧するファイル

### 最新予測

[`predictions/latest_prediction.txt`](predictions/latest_prediction.txt)

人がそのまま読める最新予測、バックテスト、モデル重み、上位候補です。

### 累積予測と当選照合

[`predictions/cumulative_results.txt`](predictions/cumulative_results.txt)

過去に実際に出した各予測を、実際の本数字と照合したテキストです。当選金額は「その購入方式で買っていた場合」の方式別払戻金として表示します。

### 全予測の機械可読履歴

[`predictions/prediction_history.csv`](predictions/prediction_history.csv)

各回・各順位の予測確率、照合状態、本数字、成立した購入方式、払戻金を累積保存します。

### 過去回別の精度監査

[`backtests/historical_accuracy.txt`](backtests/historical_accuracy.txt)

第501回以降について、各回より前のデータだけを使って予測した **未来情報リークなしの walk-forward 精度**を表示します。

確認できる指標:

- Top1 / Top3 / Top5 / Top10 / Top20 ストレート的中率
- 一様ランダム基準との比較
- 95% Wilson 信頼区間
- 実際の当選数字が1000通り中で何位だったか
- Exact NLL / 1桁 log loss と一様分布の比較
- Top20 BOX / ミニの照合率
- 直近100 / 500 / 1000回の精度
- 年別精度

回別の詳細は [`backtests/historical_rounds/`](backtests/historical_rounds/) に1,000回単位で分割保存します。各行に **回号・抽せん日・本数字・Top1・本数字の順位・Top20候補・ストレート/BOX/ミニ判定**を記録します。

### 世代ごとのバックテスト履歴

[`backtests/backtest_history.csv`](backtests/backtest_history.csv)

世代ごとの log loss、一様分布との差、Top1/Top20包含率、モデル重みを保存します。

## 過去回別 walk-forward のルール

過去精度は、現在の全データを使って過去を当て直す方式ではありません。これは精度を不当に水増しするため禁止しています。

- 最初の500回は学習用 warm-up
- 第501回から評価開始
- 第N回の予測には第N-1回までの情報だけを使用
- frequency / Markov はその時点までの履歴のみ使用
- 軽量 Logistic-loss / ExtraTrees は過去データだけで定期再学習
- 再学習区間内は直前に確定したモデルを凍結して使用
- アンサンブル重みは、当該回の結果を見る**前**の過去成績だけで決定
- 当該回の結果は次回以降の重み更新にのみ利用

このため、過去回別精度は `retrospective fit` ではなく `prequential / walk-forward` の検証値です。

## 現在の過去回別監査結果（2026-08-21・第7054回まで）

- 評価: **6,554回（第501回〜第7054回）**
- Top1: **9/6,554 = 0.1373%**（一様基準 0.100%、95%CI 0.0723%〜0.2608%）
- Top5: **35/6,554 = 0.5340%**（一様基準 0.500%）
- Top20: **135/6,554 = 2.0598%**（一様基準 2.000%、95%CI 1.7430%〜2.4328%）
- 実数字の平均順位: **499.11 / 1000**
- Exact NLL: **6.942057**（一様基準 6.907755）
- 総合判定: **予測優位性は未確認**

Top1/Top20とも一様基準が95%信頼区間内にあり、さらにExact NLLが一様分布より悪いため、現時点では持続的な予測エッジが確認できたとは判断しません。

## 当選照合の考え方

予測数字に対して以下を別々に判定します。

- **ストレート**: 3桁・並び順とも完全一致
- **ボックス**: 3桁の数字集合が一致
- **セット(ストレート)**: 完全一致時のセット・ストレート払戻
- **セット(ボックス)**: 並べ替え一致時のセット・ボックス払戻
- **ミニ**: 下2桁が一致

1つの予測数字でも購入方式によって払戻金が異なるため、単一の「当選金額」とはせず、成立する方式ごとの金額を表示します。

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

Actions の実行トークンには **生成物をリポジトリへ保存するための `contents: write`** を付与しています。リポジトリ設定・Secrets・Issues・Packages・Deployments 等への無関係な `write-all` は使用していません。これはAIエージェントが本タスクを完全自動化するために必要な書込権限を持ちつつ、漏えい時の影響範囲を最小化するためです。

## 履歴データ

全履歴は GitHub 上で `data/history_parts/part_*` に保存します。各実行時に連結して Base64 を復号し、XZ を展開して `data/numbers3.csv` を復元します。更新後は再び XZ 圧縮・Base64 化・8KB単位の分割を行い、自動コミットします。

## 手動実行

```bash
cat data/history_parts/part_* > /tmp/numbers3.csv.xz.b64
base64 -d /tmp/numbers3.csv.xz.b64 > /tmp/numbers3.csv.xz
xz -dc /tmp/numbers3.csv.xz > data/numbers3.csv

pip install -r requirements.txt

python numbers3_runner.py \
  --csv data/numbers3.csv \
  --scraper scrapingnumbers3.py \
  --refresh \
  --validation 1000 \
  --topk 20
```

過去回別監査だけを単独実行する場合:

```bash
python numbers3_historical_backtest.py \
  --csv data/numbers3.csv \
  --output-dir backtests \
  --warmup 500 \
  --refit-interval 1000 \
  --topk 20 \
  --force
```

## 注意

宝くじの当選を保証するものではありません。抽せんが独立・一様である場合、過去履歴から持続的な予測優位性を得られない可能性があります。エージェントは一様分布を基準に継続バックテストし、優位性が確認できない場合もそのまま記録します。
