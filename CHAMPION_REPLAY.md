# Champion Full-History Replay Feedback

最新Championが更新されるたびに、過去の各回を最新Champion設定で再予測します。

## 目的

- 最新Championが過去全体でどの程度の確率精度を持つか再評価する
- 第N回を予測するときは第N-1回以前のデータだけを使用し、未来情報リークを防ぐ
- 最新Championの過去回別予測を `backtests/champion_replay/latest_rounds.csv` で閲覧可能にする
- Champion世代ごとの精度を `backtests/champion_replay/replay_history.csv` に累積する
- 過去全体で改善が停滞した場合、次のChallenger探索幅を自動的に広げる

## 自動実行タイミング

1. Continuous Evolution workflow起動時
2. Championが昇格した直後
3. 30分チェックポイント後に新しい抽せんデータが取り込まれた時

同じChampion・同じデータの場合はdataset fingerprintで判定し、重い再計算をスキップします。

## 評価分離

全履歴リプレイ結果は2領域に分けます。

- **selection history**: 最後の1,000回を除いた期間。探索フィードバックに利用可能
- **final audit holdout**: 最後の1,000回。報告専用で探索方向・Champion昇格判定には使用しない

これにより「過去を何度も見て最適化した結果」を未知データ精度と誤認しないようにします。

## 主指標

- Exact NLL（主指標。低いほど良い）
- 1桁 log loss
- 当選数字の1000通り中平均順位
- Top1 / Top3 / Top5 / Top10 / Top20
- Top20 BOX / ミニ

## フィードバック

`state/champion_replay_state.json` に過去ベストselection NLLと停滞回数を保存します。

- 停滞 0〜2回: 通常探索
- 停滞 3〜5回: 2〜4パラメータを同時変更
- 停滞 6〜11回: 3〜6パラメータを同時変更
- 停滞 12回以上: 4〜7パラメータを同時変更

探索幅だけを変更し、Champion昇格のNested Walk-Forward基準は緩めません。

## 出力

- `backtests/champion_replay/latest_accuracy.txt` — 最新Championの全履歴精度サマリー
- `backtests/champion_replay/latest_rounds.csv` — 最新Championによる過去回別再予測
- `backtests/champion_replay/replay_history.csv` — Champion/データ更新ごとの精度履歴
- `state/champion_replay_state.json` — ベストNLL・停滞回数・最新評価state

## 注意

同じ過去データを繰り返し最適化するだけでは、未知の抽せんに対する精度向上は保証できません。そのためfinal audit holdoutは探索から隔離し、将来の実抽せん結果を継続的に追加して評価します。
