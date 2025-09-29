# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Categorical

# ----------------------------------------------------------------------
# 1. Transformerモデルの定義 (事前学習済みモデルの再現)
# ----------------------------------------------------------------------

class TransformerModel(nn.Module):
    """
    強化学習用のTransformerモデル。
    事前学習時に使用したモデル構造と同一である必要があります。
    """
    def __init__(self, input_dim: int, nhead: int = 4, num_layers: int = 2, d_model: int = 256):
        super().__init__()
        
        self.d_model = d_model
        
        # 入力イベントベクトルの次元をd_modelに合わせるための線形層
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, context: th.Tensor) -> th.Tensor:
        """
        :param context: (Batch Size, Sequence Length, Input Dim) の入力シーケンス
        :return: (Batch Size, Sequence Length, D_model) の出力シーケンス
        """
        # 1. 入力ベクトルの次元調整
        x = self.input_projection(context)
        
        # 2. Transformerエンコーダを適用
        # マスク処理は、麻雀の文脈ではパディング部分に対する処理のみが必要ですが、
        # ここではシンプルにするため、パディングマスクは省略します。
        transformer_output = self.transformer_encoder(x)
        
        return transformer_output

# ----------------------------------------------------------------------
# 2. 特徴抽出器 (BaseFeaturesExtractor) の定義
# ----------------------------------------------------------------------

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    観測 (Observation Dict) から特徴ベクトルを抽出する。
    ここではTransformerModelを呼び出し、シーケンスから最終的な特徴をプールする。
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # observation_spaceの'context'要素から入力次元を取得
        input_dim = observation_space["context"].shape[1] 
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)

        self.transformer = TransformerModel(input_dim=input_dim, d_model=256) # TransformerModelをインスタンス化
        
        # Transformerの出力 (D_model=256) を、Features Dim (512) に変換する層
        self.pooling_layer = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # ★事前学習済みモデルの重みロード (ここに実装)
        # try:
        #     # 例: 事前学習済み重みファイルのパスを適切に指定
        #     pretrained_weights = th.load("/content/gdrive/MyDrive/mahjong_pretrain_weights.pth")
        #     # TransformerModelに重みをロード
        #     self.transformer.load_state_dict(pretrained_weights, strict=False) 
        #     print("Pre-trained Transformer weights loaded successfully.")
        # except FileNotFoundError:
        #     print("Pre-trained weights not found. Starting RL training from scratch.")
        # except Exception as e:
        #     print(f"Error loading pre-trained weights: {e}")

    def forward(self, observations: spaces.Dict) -> th.Tensor:
        """
        :param observations: Gymnasium環境から得られる観測辞書 {'context', 'action_mask'}
        :return: (Batch Size, features_dim) の特徴ベクトル
        """
        # Contextは (Batch Size, Sequence Length, Input Dim)
        context = observations["context"] 
        
        # 1. Transformerによる処理
        transformer_output = self.transformer(context)
        
        # 2. シーケンスのプーリング (最後のイベントベクトルを採用)
        # Sequence LengthはCONTEXT_MAX_LEN (50)
        # 形状: (Batch Size, Sequence Length=50, D_model=256)
        
        # 最後のイベントベクトル (最新の状態) を利用して状態を表現
        last_event_vector = transformer_output[:, -1, :] 
        
        # 3. 最終特徴ベクトルを生成
        features = self.pooling_layer(last_event_vector)
        
        return features


# ----------------------------------------------------------------------
# 3. カスタムポリシー (ActorCriticPolicy) の定義
# ----------------------------------------------------------------------

class MaskedCategorical(Categorical):
    """
    合法的な行動のみに確率を割り当てる (Action Masking) ためのカテゴリカル分布
    """
    def __init__(self, logits: th.Tensor, mask: th.Tensor):
        # 違法な行動のlogitを非常に小さな値（例: -1e8）に設定して、確率を0にする
        masked_logits = logits.masked_fill(mask == 0, -1e8) 
        super().__init__(logits=masked_logits)
        self.mask = mask

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # PPOの学習中に呼び出される
        return super().log_prob(actions)
        
    def entropy(self) -> th.Tensor:
        # エントロピー計算
        # マスクされていない行動のみを使用してエントロピーを計算することが望ましいが、
        # Stable Baselines3の標準実装との互換性のため、ベースクラスのメソッドを使用。
        # (ただし、マスクされたlogitにより結果は合法行動に限定される)
        return super().entropy()

    def sample(self) -> th.Tensor:
        # 環境からのアクション選択時に呼び出される
        # マスクされたlogitからサンプリング
        return super().sample()


class MaskedTransformerPolicy(ActorCriticPolicy):
    """
    Transformer特徴抽出器とAction Maskingを統合したActor-Critic Policy
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule, **kwargs):
        
        # 特徴抽出器の指定
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor, # カスタム特徴抽出器を使用
            features_extractor_kwargs=kwargs.get("features_extractor_kwargs", {}),
            **kwargs
        )

        # 価値ネットワークと行動ネットワークの初期化は、ActorCriticPolicyの__init__内で
        # _build()メソッドを通じて自動的に行われる
        
        # Action Maskingのため、Custom Distributionの定義をオーバーライド
        self._action_dist = None 


    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        ActorCriticPolicyのforwardメソッドをオーバーライドし、Action Maskを処理する
        """
        # 1. 特徴抽出器による特徴ベクトルの取得 (Feature Extractor)
        # features: (Batch Size, features_dim)
        features = self.extract_features(obs, self.features_extractor)
        
        # 2. 価値関数の計算 (Value Function)
        latent_vf = self.value_net(features)
        
        # 3. 行動ネットワーク (Policy) の計算
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        
        # 4. Action Maskの取得と適用
        # obs['action_mask'] は (Batch Size, TOTAL_ACTION_DIM)
        action_mask = obs["action_mask"] 
        
        # 5. Masked Categorical分布を作成
        self._action_dist = MaskedCategorical(logits=logits, mask=action_mask)
        
        # 6. 行動のサンプリングまたは決定論的な選択
        if deterministic:
            actions = self._action_dist.mode()
        else:
            actions = self._action_dist.sample()

        return actions, latent_vf, self._action_dist.log_prob(actions)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_mask: th.Tensor = None) -> MaskedCategorical:
        """
        ActorCriticPolicyで必要なメソッド。Action Maskingに対応させる。
        """
        logits = self.action_net(latent_pi)
        
        if action_mask is None:
            # Masking情報がない場合、ここではエラーとするか、フルオープンにする
            # ただし、環境が常にobs['action_mask']を返すため、通常は発生しない
            raise ValueError("Action mask must be provided for MaskedCategorical distribution.")
            
        return MaskedCategorical(logits=logits, mask=action_mask)
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        PPOの学習時に呼び出される。
        """
        # 1. 特徴抽出と価値関数の計算
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        value = self.value_net(latent_vf)

        # 2. Action Maskの取得
        action_mask = obs["action_mask"] 

        # 3. 行動分布の取得 (Masked Categorical)
        distribution = self._get_action_dist_from_latent(latent_pi, action_mask)
        
        # 4. log_probとエントロピーの計算
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return value, log_prob, entropy
    
# ----------------------------------------------------------------------
# 4. 実行コード例 (ColabでPPOを実行するための準備)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from mahjong_rl_env.environment import MahjongEnv # 環境のインポート
    from stable_baselines3 import PPO

    # 1. 環境とモデルの準備
    env = MahjongEnv(agent_id=0)

    # 2. PPOモデルの初期化
    model = PPO(
        MaskedTransformerPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048, # 経験収集ステップ数
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tensorboard_logs/mahjong_rl/",
        device="cuda", # ColabでGPUを使用
        # features_extractor_kwargsはTransformerのハイパーパラメータを制御可能
        policy_kwargs=dict(
            features_extractor_kwargs=dict(
                features_dim=512 # 最終的な特徴次元
            ),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])] # Actor/CriticのMLP層
        )
    )

    print("---強化学習の準備完了。学習を開始します---")
    # model.learn(total_timesteps=100000)
    # model.save("mahjong_rl_agent_v1.zip")
