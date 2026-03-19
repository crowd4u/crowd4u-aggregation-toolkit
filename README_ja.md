# Crowd4U集約ツールキット

このツールキットはCrowd4Uでクラウドソーシングを行う利用者（requestor）のために、クラウドソーシング結果の品質を向上させる集約手法の実装とその利用方法を提供します。

## 集約とは？

<img width="607" height="283" alt="image" src="https://github.com/user-attachments/assets/1149ff6c-ca92-4270-ac68-24216c405803" />

クラウドソーシングでは同一のタスクを複数のワーカに割り当て、その結果を集約することで個々のワーカの誤りの影響を緩和することが一般的です。
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
