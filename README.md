pip install -r requirements.txt
generate_data.py: 天鳳ログを解析し、教師あり学習用のデータセットを生成します。
a. 事前学習済み重みのロード設定
b. 強化学習の実行
tf_agents_environment.py: TF-Agentsのフレームワークに準拠した麻雀環境のラッパー。ゲームロジックとエージェントの間のデータのやり取りを定義します。

# 🤖 Mahjong-RL-Transformer: 麻雀AI強化学習プロジェクト (TensorFlow/TF-Agents版)

本プロジェクトは、天鳳ログで事前学習させた Transformer モデルをベースに、**強化学習（Reinforcement Learning, RL）**を用いて自己対戦学習を行うことで、日本式リーチ麻雀における最高レベルの戦略獲得を目指します。

---

## 🛠️ 技術スタック

| カテゴリ         | コンポーネント         | 目的                                      |
|------------------|-----------------------|-------------------------------------------|
| MLフレームワーク | TensorFlow            | 事前学習・強化学習のモデル構築・学習      |
| RLライブラリ     | TF-Agents             | 強化学習アルゴリズム (PPO) の実行         |
| AIモデル         | Transformer           | 状況判断と行動決定の中核                  |
| 麻雀計算エンジン | mahjong               | 点数計算、向聴数計算など                  |
| 実行環境         | ローカル(GPU)/Colab   | 高速な学習のための環境                    |

---

## 📁 ディレクトリ構成

```text
mahjong-rl-transformer/
├── data/
│   ├── tenhou_logs/             # 天鳳の生ログ (.mjlog.gz)
│   └── processed_data/          # 教師あり学習データ (.pkl)
├── mahjong_rl_env/
│   ├── tf_agents_environment.py # TF-Agents互換の麻雀環境
│   └── mahjong_game_logic.py    # 純粋な麻雀のゲーム進行ロジック
├── models/
│   └── tf_agents_policy.py      # TF-Agents用Transformerポリシーネットワーク
├── notebooks/
│   └── (Colabでの実行用ノートブックなど)
├── vectorizer.py                # 教師あり/強化学習 共通のデータベクトル化
├── train_transformer.py         # [ステップ1] 天鳳ログでモデルを事前学習
├── train_rl_tf_agents.py        # [ステップ2] 強化学習でモデルを自己対戦強化
├── requirements.txt             # 依存ライブラリ一覧
└── README.md
```

---

## 🚀 セットアップと学習手順

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. 事前学習 (Supervised Learning)

1. 天鳳の対戦ログを `data/tenhou_logs/` に配置
2. 教師あり学習用データセット生成: `generate_data.py`
3. Transformerモデルの学習: `train_transformer.py`

### 3. 強化学習 (Reinforcement Learning)

1. 事前学習済みモデルの重みを `train_rl_tf_agents.py` でロード
    ```python
    # train_rl_tf_agents.py
    PRETRAINED_MODEL_PATH = 'path/to/your/pretrained_transformer.keras' # 事前学習済みモデルのパス
    if os.path.exists(PRETRAINED_MODEL_PATH):
        actor_critic_net.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
        print(f"✅ Pre-trained weights loaded from {PRETRAINED_MODEL_PATH}")
    ```
2. 強化学習の実行
    ```bash
    python train_rl_tf_agents.py
    ```

---

## ⚙️ 主要モジュールの機能概要

### vectorizer.py
教師あり学習・強化学習で同一の観測データを生成する共通モジュール。ゲームイベント履歴をTransformerが解釈できる固定長ベクトルに変換。

### mahjong_rl_env/
- `tf_agents_environment.py`: TF-Agents準拠の麻雀環境ラッパー。ゲームロジックとエージェント間のデータやり取りを定義。
- `mahjong_game_logic.py`: 純粋な麻雀ルール・状態遷移・報酬計算などを担うクラス。

### models/tf_agents_policy.py
`MahjongActorCriticNetwork`: TF-Agents流のTransformerベースネットワーク。事前学習モデルと同じ構造で、方策（Actor）と状態価値（Critic）を出力。Action Masking対応。