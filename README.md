# 🤖 Mahjong-RL-Transformer: 麻雀 AI 強化学習プロジェクト

本プロジェクトは、**天鳳ログ**で事前学習させた **Transformer モデル**をベースに、**強化学習（Reinforcement Learning, RL）**を用いて自己対戦学習を行うことで、**日本式リーチ麻雀**における最高レベルの戦略獲得を目指すことを目的としています。

強化学習環境には **Gymnasium** を採用し、牌理計算には高精度な **`mahjong` ライブラリ**を利用しています。学習は **Google Colab** の GPU 環境で効率的に実行できるように設計されています。

---

## 🛠️ 技術スタック

| カテゴリ | コンポーネント | 目的 |
| :--- | :--- | :--- |
| **RL フレームワーク** | `Stable Baselines3 (PPO)` | 強化学習アルゴリズムの実行 |
| **AI モデル** | `PyTorch` / `Transformer` | 事前学習済み知識の活用と行動決定 |
| **環境インターフェース** | `Gymnasium` | RL エージェントと環境の連携 |
| **麻雀計算エンジン** | `mahjong` (v1.3.0) | 点数計算、向聴数計算など |
| **実行環境** | `Google Colab (GPU)` | 高速な学習のための環境 |

---

## 📁 ディレクトリ構成

プロジェクトのモジュール化と管理を容易にするための構造です。

mahjong-rl-transformer/
├── data/
│   ├── tenhou_logs/             # 天鳳の生ログ (.mjlog.gz)
│   └── processed_data/          # 教師あり学習データ (.pkl)
├── mahjong_rl_env/
│   ├── environment.py           # MahjongEnv (Gymnasium互換の4人AI対戦環境)
│   ├── feature_converter.py     # 観測情報 (Context/Mask) のベクトル化ロジック
│   └── rule_checker.py          # (オプション) 行動の合法性チェックの分離
├── models/
│   ├── transformer_model.py     # Transformer モデル構造の定義
│   └── custom_policy.py         # MaskedTransformerPolicy (Action MaskingとRL統合)
├── notebooks/
│   └── Colab_RL_Setup.ipynb     # Colabでの学習実行用ノートブック (メイン実行ファイル)
├── scripts/                     # データ準備、事前学習、RL学習のシェルスクリプト
├── .gitignore
└── requirements.txt             # 依存ライブラリ一覧


---

## 🚀 セットアップと実行手順

### 1. 依存ライブラリのインストール

ローカル環境または Colab の最初のセルで実行します。

```bash
# Colab での実行を推奨
!pip install mahjong gymnasium stable-baselines3[extra] torch numpy
2. Google Drive のセットアップ (Colab 必須)
学習済みモデルの永続化と、自作モジュールへのアクセス設定です。

notebooks/Colab_RL_Setup.ipynb を Colab で開きます。

ノートブック内の手順に従い、Google Drive をマウントし、PROJECT_ROOT パスを設定します。

本リポジトリの全ファイルを、設定した PROJECT_ROOT ディレクトリ内に配置します。

3. 事前学習済み重みのロード
RL 学習を開始する前に、Transformer モデルに天鳳ログで獲得した知識を注入します。

ファイル: models/custom_policy.py

場所: TransformerFeatureExtractor クラスの __init__ メソッド内

Python

        # ★事前学習済みモデルの重みロード (ここに実装)
        try:
            # 事前学習済み重みファイルのパスを適切に指定
            pretrained_weights = th.load("/content/gdrive/MyDrive/mahjong-rl-transformer/pretrain_weights.pth")
            self.transformer.load_state_dict(pretrained_weights, strict=False) 
            print("Pre-trained Transformer weights loaded successfully.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Starting RL training from scratch.")
4. 強化学習の実行
Colab_RL_Setup.ipynb ノートブックを実行します。

ノートブックは以下の処理を自動で行います。

MahjongEnv (4人AIセルフプレイ環境) の初期化。

MaskedTransformerPolicy (カスタムポリシー) をPPOに組み込み。

GPU を利用して学習を開始し、TensorBoard でログを記録。

学習終了時、または中断時にモデルを自動保存。

⚙️ 主要モジュールの機能概要
mahjong_rl_env/environment.py
セルフプレイ: step メソッド内で、RL エージェントの番が来るまで他家（Bot）に自動で行動を選択させるロジック (_process_opponent_turns) を実装。

報酬計算: 和了・放銃・流局時に mahjong.HandCalculator を使用し、点数移動を正確に報酬 (reward) として返す。

観測構造: {"context": ..., "action_mask": ...} の Dict 構造を返す。

models/custom_policy.py
TransformerFeatureExtractor: 環境から受け取った時系列データ (context) を Transformer に通し、現在の状態を表す固定長の特徴ベクトルを抽出。

MaskedCategorical: 違法な行動の Logit をマスク (-1e8) し、RL エージェントが常に合法的な行動のみを選択するように強制する。