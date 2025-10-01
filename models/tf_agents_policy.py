# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tf_agents.networks import network

# プロジェクトルートからモジュールをインポート
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorizer import CONTEXT_MAX_LEN, EVENT_VECTOR_DIM

class MahjongActorCriticNetwork(network.Network):
    """
    Transformerベースのアクター・クリティックネットワーク。
    事前学習済みモデルの構造と互換性を持たせる。
    """
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 num_heads=4,
                 d_model=128,
                 num_transformer_blocks=2,
                 name='MahjongActorCriticNetwork'):
        super(MahjongActorCriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )
        self._output_tensor_spec = output_tensor_spec
        self.num_actions = output_tensor_spec.shape[-1]

        # --- Transformer Encoder (train_transformer.py のモデル構造と一致させる) ---
        self.input_projection = layers.Dense(d_model, activation='relu', name="input_projection")
        self.pos_embedding = layers.Embedding(input_dim=CONTEXT_MAX_LEN, output_dim=d_model, name="pos_embedding")

        self.transformer_blocks = []
        for i in range(num_transformer_blocks):
            self.transformer_blocks.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name=f"mha_{i}"),
                layers.LayerNormalization(epsilon=1e-6, name=f"norm1_{i}"),
                layers.Dense(d_model*2, activation='relu', name=f"ffn1_{i}"),
                layers.Dense(d_model, name=f"ffn2_{i}"),
                layers.LayerNormalization(epsilon=1e-6, name=f"norm2_{i}"),
            ])
        self.pooling = layers.GlobalAveragePooling1D()

        # --- Actor (Policy) と Critic (Value) のヘッド ---
        self.actor_head = layers.Dense(self.num_actions, name='actor_head')
        self.critic_head = layers.Dense(1, name='critic_head')

    def call(self, observations, step_type=(), network_state=()):
        context = observations['context']
        mask = observations['action_mask']

        # --- Transformerによる特徴抽出 ---
        positions = tf.range(start=0, limit=CONTEXT_MAX_LEN, delta=1)
        pos_emb = self.pos_embedding(positions)
        x = self.input_projection(context) + pos_emb

        for mha, norm1, ffn1, ffn2, norm2 in self.transformer_blocks:
            attn_output = mha(x, x)
            x = norm1(x + attn_output)
            ffn_output = ffn2(ffn1(x))
            x = norm2(x + ffn_output)

        features = self.pooling(x)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)

        # --- Action Masking ---
        masked_logits = action_logits + (tf.cast(mask, tf.float32) - 1.0) * 1e8

        return (masked_logits, tf.squeeze(value, -1)), network_state

