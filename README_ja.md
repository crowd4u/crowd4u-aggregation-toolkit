# Crowd4U集約ツールキット

このツールキットはCrowd4Uでクラウドソーシングを行う利用者（requestor）のために、クラウドソーシング結果の品質を向上させる集約手法の実装とその利用方法を提供します。

## 集約とは？

<img width="607" height="283" alt="image" src="https://github.com/user-attachments/assets/1149ff6c-ca92-4270-ac68-24216c405803" />

マルチクラス分類を扱うクラウドソーシングの後処理では同一のタスクを複数のワーカに割り当て、その結果を集約することで個々のワーカの誤りの影響を緩和することが一般的です。
最も簡単な集約手法は多数決ですが、クラウドソーシングではワーカ間の能力差が大きく、能力の低いワーカが多数派であることも多いため、多数決が最適であるとは限りません。

<img width="610" height="265" alt="image" src="https://github.com/user-attachments/assets/83487873-fdeb-49aa-a4a6-c33eb2203f39" />

そのため、潜在クラスモデルを用いた教師無し学習で、ワーカの能力を推定し、その推定結果を用いて重み付き多数決を行う手法がスタンダートとなっています。

このリポジトリでは多数決に加えて、これらの教師なし学習を用いた集約手法の実装を提供します。
さらに、このリポジトリではCrowd4Uの特有の機能である、「AIワーカ」に対応した集約手法も提供します。

## 利用方法
### 1. Docker実行環境の構築
このツールキットでは、環境構築を容易にするために、仮想コンテナ環境であるDockerを利用します。
はじめに、お使いのコンピュータ（Windows/Mac/Linux)にDocker実行環境をセットアップしてください。

インストール方法に関してはDockerのマニュアルをご確認ください。
  - Windows: https://docs.docker.jp/desktop/install/windows-install.html
  - Mac (Apple Silicon): https://docs.docker.jp/desktop/mac/apple-silicon.html
  - Linux: https://docs.docker.jp/desktop/install/linux-install.html

※Apple SiliconなどのARMアーキテクチャのコンピュータでは動作確認をしておりません。`x64`アーキテクチャでの実行を推奨します。

### 2. ツールキットをダウンロードする
#### `git`コマンドが利用可能な場合
このリポジトリをクローンしてください。
#### `git`コマンドが利用できない・わからない場合
<img width="500" height="252" alt="image" src="https://github.com/user-attachments/assets/9a103df4-5e7d-49f2-9d3b-e4acab1b7d44" />

1. https://github.com/crowd4u/crowd4u-aggregation-toolkit/ にアクセスします。
2. ページ右上の緑色「Code」ボタン（写真赤丸部分）を押して、「Download ZIP」を押します。
3. ダウンロードしたZIPファイルを展開し、任意の場所に保存してください。

### 3. クラウドソーシング結果データを加工・保存する
このツールキットでは、以下のような形式のCSVファイルのみを入力データとして受け取ります。
また、AIワーカを利用する場合は、人間ワーカの結果と別々のCSVで保存する必要があります。
```csv
task,worker,label
question1,worker1,dog
question1,worker2,cat
question1,worker3,dog
question2,worker1,dog
question2,worker2,hamster
question2,worker3,hamster
question3,worker1,parrot
question3,worker2,parrot
question3,worker3,cat
```

 - `task`: タスクID
 - `worker`: ワーカID
 - `label` : 回答のラベル

加えて、ワーカが回答する可能性のある`label`の一覧を別途、jsonファイルで保存する必要があります。
```json
[
    "dog",
    "cat",
    "rabbit",
    "hamster",
    "parrot"
]
```

**これらのファイルはこのツールキットの`datasets/`配下に保存される必要があります。**
`datasets/`にはサンプルデータが入っていますので、こちらも併せてご確認ください。

**なお、このツールキットはCrowd4UのTARテーブルデータからツールキット用の形式に変換する操作を支援するノートブックを提供しています。TAR形式から変換する場合は、以下のノートブックもご覧ください。また、このようなデータ変換は生成AI等でも容易に行えます。**

- wip

### 4. 集約を実行する
1. `docker`コマンドが利用可能なターミナルを起動します。
2. ターミナルでツールキットのフォルダへ移動します。
3. 以下のコマンドを実行し、Dockerコンテナを起動します（初回実行時は起動に時間がかかります）
```sh
docker compose up -d
```
4. 起動が完了したら、以下のコマンドで集約を実行します。
```sh
docker exec crowd4u-aggregation-toolkit python main.py <集約手法名> <出力ファイル名> <ラベル一覧のJSONファイル名> <人間データのCSVファイル名> <AIデータのCSVファイル名（省略可能）>
```
 - `<集約手法名>` 集約に用いる手法を指定します。以下の手法が指定できます。
    - `MV` : 単純多数決。[Crowd-Kit](https://github.com/Toloka/crowd-kit)の実装を利用しています。
    - `DS` :  Dawid-Skene法 [(Dawid & Skene 1979)](https://doi.org/10.2307/2346806)。Crowd-Kitの実装を利用しています。
    - `GLAD` : GLAD法 [(Whitehill et al. 2009)](https://dl.acm.org/doi/10.5555/2984093.2984321)。Crowd-Kitの実装を利用しています。
    - `BDS` : マルコフ連鎖モンテカルロ法（MCMC）を用いたBayesian Dawid-Skene法 [(Paun et al. 2018)](https://aclanthology.org/Q18-1040/)。初回実行時のみ時間がかかります。
    - `HSDS_EM` : AIの不均衡な能力の問題に対処するために提案されたHuman-Seeded Dawid-Skene (HS-DS) 法 (our paper; under review) のEMアルゴリズムによる実装。人間のみのデータには適用できません。サンプルデータのようなかなり小規模なデータではエラーが発生し利用できないことがあります。
    - `HSDS_MCMC` : AIの不均衡な能力の問題に対処するために提案されたHS-DS法のMCMCによる実装。人間のみのデータには適用できません。初回実行時のみ時間がかかります。
- `<出力ファイル名>` : 集約結果データのファイル名を指定します。集約結果は`results/`に保存されます。
- `<ラベル一覧のJSONファイル名>` : 「3. クラウドソーシング結果データを加工・保存する」で作成したJSONファイルを指定します。
- ` <人間データのCSVファイル名>` : 「3. クラウドソーシング結果データを加工・保存する」で作成した人間データのCSVファイルを指定します。
- ` <AIデータのCSVファイル名（省略可能）>` : 「3. クラウドソーシング結果データを加工・保存する」で作成したAIデータのCSVファイルを指定します。AIデータが存在しない場合は指定不要です。人間ワーカとAIワーカを区別しない集約手法（HS-DS以外）の場合は、人間データと縦結合されてデータ処理されます。

例えば、以下のように実行できます。
```sh
docker exec crowd4u-aggregation-toolkit python main.py MV sample_result.csv sample_labels.json sample_human.csv 
```
```sh
docker exec crowd4u-aggregation-toolkit python main.py DS  sample_result.csv sample_labels.json sample_human.csv sample_ai.csv
```
```sh
docker exec crowd4u-aggregation-toolkit python main.py HSDS_MCMC  sample_result.csv sample_labels.json sample_human.csv sample_ai.csv
```
5. 集約結果は`results/`に指定した名前で保存されます。
6. コンテナの実行を終了するには、以下のコマンドを実行します。
```sh
docker compose stop
```
## このツールキットの貢献者
このツールキットの開発に貢献しているのは以下のメンバーです。

- Takumi TAMURA : https://takumi1001.github.io/takumi1001/

ツールキットをはじめCrowd4Uについてのお問い合わせは、以下の連絡先をご確認ください。

https://fusioncomplab.org/about.html


