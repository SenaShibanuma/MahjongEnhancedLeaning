# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks import network
import os
import json
from collections import Counter
import asyncio
import websockets
from pyngrok import ngrok, conf
import sys
import numpy as np

# 既存のモジュールをインポート
from mahjong_rl_env.tf_agents_environment import MahjongPyEnvironment


# --- ハイパーパラメータ ---
NUM_ITERATIONS = 500_000
COLLECT_EPISODES_PER_ITERATION = 1
REPLAY_BUFFER_CAPACITY = 2048
LEARNING_RATE = 3e-5
LOG_INTERVAL = 100
CHECKPOINT_DIR = './rl_checkpoints'
KIFU_DIR = './rl_kifu'
KIFU_SAVE_INTERVAL = 5000
PRETRAINED_MODEL_PATH = './models/senas_jan_ai_transformer_v1.keras'
WEBSOCKET_PORT = 8765


# === ▼▼▼ 新しいネットワーク定義 (最終修正版) ▼▼▼ ===

class SharedEncoderModel(tf.keras.Model):
    """Transformerベースの共有エンコーダをカプセル化するKerasモデル"""
    def __init__(self, num_transformer_blocks=2, d_model=128, num_heads=4, ff_dim=256, name="SharedEncoder"):
        super(SharedEncoderModel, self).__init__(name=name)
        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append([
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
                layers.Add(),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(d_model),
                layers.Add(),
                layers.LayerNormalization(epsilon=1e-6)
            ])
        self.global_avg_pool = layers.GlobalAveragePooling1D()

    def call(self, inputs, training=False):
        x = inputs
        for (attn, add1, norm1, ffn1, ffn2, add2, norm2) in self.transformer_blocks:
            attn_output = attn(query=x, value=x, key=x, training=training)
            x_res = add1([x, attn_output])
            x_norm = norm1(x_res, training=training)
            ffn_output = ffn2(ffn1(x_norm, training=training), training=training)
            x_res2 = add2([x_norm, ffn_output])
            x = norm2(x_res2, training=training)
        return self.global_avg_pool(x)

class ActorNet(network.Network):
    def __init__(self, input_tensor_spec, action_spec, shared_encoder, name='ActorNet'):
        super(ActorNet, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
        self._shared_encoder = shared_encoder
        if action_spec.shape.rank == 0:
            self._num_actions = int(action_spec.maximum - action_spec.minimum + 1)
        else:
            self._num_actions = action_spec.shape.dims[-1]
        self._actor_head = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self._num_actions, name='action_logits')
        ])

    # === ▼▼▼ 修正箇所: ActorNetにも出力仕様を明記 (最重要) ▼▼▼ ===
    @property
    def output_tensor_spec(self):
        return tf.TensorSpec(shape=(self._num_actions,), dtype=tf.float32)

    def call(self, observations, step_type=(), network_state=(), training=False):
        obs_vector = observations['observation']
        action_mask = observations['action_mask']
        
        encoded_state = self._shared_encoder(obs_vector, training=training)
        action_logits = self._actor_head(encoded_state, training=training)
        
        masked_logits = tf.where(tf.cast(action_mask, tf.bool), action_logits, -1e8)
        return masked_logits, network_state

class ValueNet(network.Network):
    def __init__(self, input_tensor_spec, shared_encoder, name='ValueNet'):
        super(ValueNet, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
        self._shared_encoder = shared_encoder
        self._value_head = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, name='value_prediction')
        ])
    
    @property
    def output_tensor_spec(self):
        return tf.TensorSpec(shape=(), dtype=tf.float32)

    def call(self, observations, step_type=(), network_state=(), training=False):
        obs_vector = observations['observation']
        encoded_state = self._shared_encoder(obs_vector, training=training)
        value_prediction = self._value_head(encoded_state, training=training)
        return tf.squeeze(value_prediction, axis=-1), network_state

# --- WebSocketサーバー関連 (変更なし) ---
CONNECTED_CLIENTS = set()
async def broadcast(message):
    if CONNECTED_CLIENTS:
        await asyncio.wait([client.send(message) for client in CONNECTED_CLIENTS])
async def websocket_handler(websocket, path):
    CONNECTED_CLIENTS.add(websocket)
    print(f"Client connected. Total clients: {len(CONNECTED_CLIENTS)}")
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"Client disconnected. Total clients: {len(CONNECTED_CLIENTS)}")
async def start_server():
    NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        print("✅ ngrok authtoken configured.")
    else:
        print("⚠️ ngrok authtoken not found...")
    if 'google.colab' in sys.modules:
        conf.get_default().region = 'jp'
    server = await websockets.serve(websocket_handler, "localhost", WEBSOCKET_PORT)
    print(f"WebSocket server started on ws://localhost:{WEBSOCKET_PORT}")
    try:
        tunnel = ngrok.connect(WEBSOCKET_PORT, "tcp")
        public_url = tunnel.public_url.replace("tcp://", "ws://")
        print("====================================================================================")
        print(f"🚀 Dashboard URL: {public_url}")
        print("====================================================================================")
        await asyncio.Future()
    except Exception as e:
        print(f"❌ Failed to start ngrok tunnel: {e}")

async def main_training_loop():
    py_env = MahjongPyEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    shared_encoder = SharedEncoderModel()
    actor_net = ActorNet(tf_env.observation_spec(), tf_env.action_spec(), shared_encoder)
    value_net = ValueNet(tf_env.observation_spec(), shared_encoder)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    agent = ppo_agent.PPOAgent(
        time_step_spec=tf_env.time_step_spec(), action_spec=tf_env.action_spec(),
        actor_net=actor_net, value_net=value_net, optimizer=optimizer, num_epochs=10,
    )
    agent.initialize()

    train_step_counter = tf.Variable(0, name="train_step_counter")
    checkpoint = tf.train.Checkpoint(
        shared_encoder=shared_encoder, actor_head=actor_net._actor_head,
        value_head=value_net._value_head, optimizer=optimizer,
        train_step_counter=train_step_counter
    )
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)
    if checkpoint_manager.latest_checkpoint:
        print(f"Restoring from checkpoint: {checkpoint_manager.latest_checkpoint}")
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")
    os.makedirs(KIFU_DIR, exist_ok=True)
    yaku_counter = Counter()
    total_rewards = []
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=REPLAY_BUFFER_CAPACITY
    )
    agent.train = common.function(agent.train)
    time_step = tf_env.reset()
    
    print("\n--- Starting Reinforcement Learning Training ---")
    for i in range(NUM_ITERATIONS):
        final_time_step, _ = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_episodes=COLLECT_EPISODES_PER_ITERATION
        ).run(time_step)
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        replay_buffer.clear()
        time_step = final_time_step
        step = agent.train_step_counter.numpy()
        last_episode_yaku = py_env.get_info().get('last_yaku_result', [])
        if last_episode_yaku: yaku_counter.update(last_episode_yaku)
        reward_value = final_time_step.reward.numpy()[0]
        total_rewards.append(reward_value)
        if i > 0 and i % LOG_INTERVAL == 0:
            avg_reward = np.mean(total_rewards)
            summary_data = {"step": int(step), "loss": float(train_loss.loss.numpy()), "avg_reward": float(avg_reward), "yaku_summary": yaku_counter.most_common(10)}
            await broadcast(json.dumps(summary_data))
            print(f"Step: {step}, Avg Reward (last {len(total_rewards)} games): {avg_reward:.2f}, Loss: {summary_data['loss']:.4f}")
            total_rewards = []
        if step > 0 and step % KIFU_SAVE_INTERVAL == 0:
            kifu_path = os.path.join(KIFU_DIR, f"kifu_step_{step}.json")
            with open(kifu_path, 'w', encoding='utf-8') as f:
                json.dump(py_env.get_info()['game_events'], f, ensure_ascii=False, indent=2)
            print(f"🀄 Kifu saved to {kifu_path}")
        if step > 0 and step % 1000 == 0:
            checkpoint_manager.save()
            print(f"💾 Checkpoint saved at step {step}")

async def main():
    server_task = asyncio.create_task(start_server())
    training_task = asyncio.create_task(main_training_loop())
    done, pending = await asyncio.wait([server_task, training_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending: task.cancel()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Shutting down ngrok...")
        ngrok.kill()

