# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tf_agents.networks import network

class MahjongActorCriticNetwork(network.Network):
    """
    TF-Agents PPOと連携するための、TransformerベースのActor-Criticネットワーク。
    """
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec, # これが環境の action_spec
                 d_model=128,
                 num_heads=4,
                 ff_dim=256,
                 num_transformer_blocks=2,
                 actor_fc_layers=(256, 256),
                 value_fc_layers=(256, 256),
                 name='MahjongActorCriticNetwork'):

        super(MahjongActorCriticNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        # 観測データの仕様を分解
        observation_spec = input_tensor_spec['observation']
        
        # --- 最終修正 ---
        # TensorShapeが空(rank=0)かどうかの判定を、TensorFlowの正しい作法に修正
        if output_tensor_spec.shape.rank == 0:
            self.num_actions = int(output_tensor_spec.maximum - output_tensor_spec.minimum + 1)
        else:
            self.num_actions = output_tensor_spec.shape.dims[-1]
        # --- 修正ここまで ---

        # ★★★ 修正箇所 ① ★★★
        # 初期化時に受け取った変数をインスタンス変数として保存
        self.num_transformer_blocks = num_transformer_blocks
        # ★★★★★★★★★★★★★

        # === 共通のTransformerエンコーダ部分 ===
        self.encoder_layers = []
        for _ in range(self.num_transformer_blocks): # self. を付けて参照
            self.encoder_layers.extend([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
                layers.Add(),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
                layers.Add(),
                layers.LayerNormalization(epsilon=1e-6),
            ])
        self.global_avg_pool = layers.GlobalAveragePooling1D()

        # === Actor (行動決定) ネットワーク ===
        self.actor_layers = [layers.Dense(units, activation='relu') for units in actor_fc_layers]
        self.action_head = layers.Dense(self.num_actions, name='action_logits')

        # === Critic (価値評価) ネットワーク ===
        self.value_layers = [layers.Dense(units, activation='relu') for units in value_fc_layers]
        self.value_head = layers.Dense(1, name='value_prediction')

    def call(self, observations, step_type=(), network_state=(), training=False):
        # 観測データとアクションマスクを取得
        obs_vector = observations['observation']
        action_mask = observations['action_mask']

        # --- 1. 共通エンコーダ ---
        current_input = obs_vector
        # ★★★ 修正箇所 ② ★★★
        # self.num_transformer_blocks を使ってループする
        for i in range(self.num_transformer_blocks):
        # ★★★★★★★★★★★★★
            # Multi-Head Attention
            attn_layer = self.encoder_layers[i * 7]
            add_layer1 = self.encoder_layers[i * 7 + 1]
            norm_layer1 = self.encoder_layers[i * 7 + 2]
            
            attn_output = attn_layer(current_input, current_input)
            x1 = add_layer1([current_input, attn_output])
            x1_norm = norm_layer1(x1)

            # Feed Forward Network
            ffn_dense1 = self.encoder_layers[i * 7 + 3]
            ffn_dense2 = self.encoder_layers[i * 7 + 4]
            add_layer2 = self.encoder_layers[i * 7 + 5]
            norm_layer2 = self.encoder_layers[i * 7 + 6]
            
            ffn_output = ffn_dense1(x1_norm)
            ffn_output = ffn_dense2(ffn_output)
            x2 = add_layer2([x1_norm, ffn_output])
            current_input = norm_layer2(x2)
        
        # シーケンス全体の特徴を要約
        encoded_state = self.global_avg_pool(current_input)

        # --- 2. Actor Head ---
        actor_features = encoded_state
        for layer in self.actor_layers:
            actor_features = layer(actor_features)
        action_logits = self.action_head(actor_features)
        
        # アクションマスクを適用
        masked_logits = tf.where(tf.cast(action_mask, tf.bool), action_logits, -1e8)

        # --- 3. Critic Head ---
        value_features = encoded_state
        for layer in self.value_layers:
            value_features = layer(value_features)
        value_prediction = self.value_head(value_features)

        # === ▼▼▼ 変更箇所 ▼▼▼ ===
        # TF-Agentsが期待する形状 (バッチサイズ,) に合わせるため、最後の次元を削除
        value_prediction = tf.squeeze(value_prediction, axis=-1)
        # === ▲▲▲ 変更ここまで ▲▲▲ ===

        return (masked_logits, value_prediction), network_state
