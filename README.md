# 🤖 Mahjong-RL-Transformer: 麻雀AI強化学習プロジェクト (TensorFlow/TF-Agents版)

本プロジェクトは、天鳳ログで事前学習させた Transformer モデルをベースに、**強化学習（Reinforcement Learning, RL）**を用いて自己対戦学習を行うことで、日本式リーチ麻雀における最高レベルの戦略獲得を目指すものです。学習フレームワークを TensorFlow と TF-Agents に統一し、モデル変換のオーバーヘッドなくシームレスな学習パイプラインを実現しています。

---

## ✨ 新機能：リアルタイム学習ダッシュボード

Webブラウザ上で、AIの学習状況をリアルタイムに監視し、保存された対局棋譜を再生することができます。

- **リアルタイムステータス**: 学習ステップ、平均報酬、損失（Loss）、和了役のランキングなどを動的に表示。
- **牌譜ビューア**: 学習の途中で自動保存された対局棋譜（JSON）を読み込み、一手ずつ再生・分析が可能。
- **Colab対応**: `ngrok`との連携により、Google Colabで実行中の学習サーバーにもローカルのブラウザから接続できます。

---

## 🛠️ 技術スタック

| カテゴリ | コンポーネント | 目的 |
| :--- | :--- | :--- |
| ML フレームワーク | TensorFlow | 事前学習および強化学習のモデル構築・学習 |
| RL ライブラリ | TF-Agents | 強化学習アルゴリズム (PPO) の実行と環境管理 |
| リアルタイム通信 | websockets, pyngrok | 学習進捗のリアルタイムGUI表示 |
| AI モデル | Transformer | 状況判断と行動決定の中核 |
| 麻雀計算エンジン | `mahjong` | 点数計算、向聴数計算など |
| 実行環境 | ローカル (GPU) / Google Colab | 高速な学習のための環境 |

---

## 📁 ディレクトリ構成

```text
mahjong-rl-transformer/
├── data/                      # 学習データ (天鳳ログなど)
├── rl_checkpoints/            # 強化学習モデルのチェックポイント
├── rl_kifu/                   # AIの対局棋譜 (.json)
├── mahjong_rl_env/
│   ├── tf_agents_environment.py   # TF-Agents互換の麻雀環境
│   └── mahjong_game_logic.py      # 純粋な麻雀のゲーム進行ロジック
├── models/
│   └── tf_agents_policy.py        # TF-Agents用Transformerポリシーネットワーク
├── notebooks/                 # 実験・分析用Jupyter Notebook
├── vectorizer.py              # 共通のデータベクトル化モジュール
├── train_transformer.py       # [ステップ1] 天鳳ログでモデルを事前学習
├── train_rl_tf_agents.py      # [ステップ2] 強化学習でモデルを自己対戦強化
├── dashboard.html             # ローカル実行用のダッシュボード
├── dashboard-colab.html       # Colab接続用のダッシュボード
├── requirements.txt           # 依存ライブラリ一覧
└── README.md
```

---

## 🚀 セットアップと学習手順

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. 事前学習 (Supervised Learning)

天鳳の対戦ログ (`.mjlog`) を `data/tenhou_logs/` に配置し、`generate_data.py` と `train_transformer.py` を実行して、事前学習済みモデル (`.keras`) を生成します。

### 3. 強化学習の実行と監視

`train_rl_tf_agents.py` を実行して強化学習を開始します。実行環境に応じて監視方法を選択してください。

#### A) ローカル環境での実行

1.  **学習開始:**

    ```bash
    python train_rl_tf_agents.py
    ```

2.  **ダッシュボード表示:**
    PCのWebブラウザで `dashboard.html` ファイルを開きます。自動的に接続され、リアルタイムで進捗が表示されます。

#### B) Google Colab 環境での実行

1.  **ngrokの準備 (初回のみ):**
    - [ngrok公式サイト](https://ngrok.com/)でアカウントを登録し、認証トークンをコピーします。
    - Colabノートブックの「シークレット」(鍵アイコン)に `NGROK_AUTH_TOKEN` という名前でトークンを登録します。

2.  **学習開始:**
    Colabのセルで、プロジェクトディレクトリに移動し、スクリプトを実行します。

    ```bash
    !python train_rl_tf_agents.py
    ```

3.  **ダッシュボード接続:**
    - 実行ログに表示される `ws://....ngrok.io` から始まる公開URLをコピーします。
    - PCのWebブラウザで `dashboard-colab.html` ファイルを開きます。
    - コピーしたURLを入力欄に貼り付け、「接続」ボタンをクリックします。

---

## ⚙️ 主要モジュールの機能概要

- **`vectorizer.py`**:
  教師あり学習と強化学習で完全に同一の観測データを生成するための共通モジュール。
- **`mahjong_rl_env/`**:
  TF-Agentsのフレームワークに準拠した麻雀環境と、独立したゲームロジックを提供。
- **`models/tf_agents_policy.py`**:
  TF-Agentsの作法に沿って定義されたTransformerベースのネットワーク。
- **`train_rl_tf_agents.py`**:
  学習のメインループに加え、WebSocketサーバーとngrok連携機能を持ち、ダッシュボードへリアルタイムにデータを送信します。