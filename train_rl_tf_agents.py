# -*- coding: utf-8 -*-
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
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
from models.tf_agents_policy import MahjongActorCriticNetwork

# --- ハイパーパラメータ ---
NUM_ITERATIONS = 500_000
COLLECT_EPISODES_PER_ITERATION = 1
REPLAY_BUFFER_CAPACITY = 2048
LEARNING_RATE = 3e-5
LOG_INTERVAL = 100
CHECKPOINT_DIR = './rl_checkpoints'
KIFU_DIR = './rl_kifu'
KIFU_SAVE_INTERVAL = 5000
# ★★★ 事前学習済みモデルのパスを更新 ★★★
PRETRAINED_MODEL_PATH = './models/senas_jan_ai_transformer_v1.keras'
WEBSOCKET_PORT = 8765

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
    """WebSocketサーバーとngrokトンネルを起動する"""
    NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        print("✅ ngrok authtoken configured.")
    else:
        print("⚠️ ngrok authtoken not found. Please set the NGROK_AUTH_TOKEN environment variable.")
        print("   You can get a token from https://dashboard.ngrok.com/get-started/your-authtoken")

    if 'google.colab' in sys.modules:
        conf.get_default().region = 'jp'

    server = await websockets.serve(websocket_handler, "localhost", WEBSOCKET_PORT)
    print(f"WebSocket server started on ws://localhost:{WEBSOCKET_PORT}")

    try:
        tunnel = ngrok.connect(WEBSOCKET_PORT, "tcp")
        public_url = tunnel.public_url.replace("tcp://", "ws://")
        print("====================================================================================")
        print("🚀 Dashboard is accessible from your local browser!")
        print(f"   Copy this WebSocket URL: {public_url}")
        print("   Paste it into 'dashboard-colab.html' to connect.")
        print("====================================================================================")
        await server.wait_closed()
    except Exception as e:
        print(f"❌ Failed to start ngrok tunnel: {e}")


async def main_training_loop():
    """メインの学習ループ"""
    py_env = MahjongPyEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    actor_critic_net = MahjongActorCriticNetwork(
        tf_env.observation_spec(), tf_env.action_spec()
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    agent = ppo_agent.PPOAgent(
        time_step_spec=tf_env.time_step_spec(), action_spec=tf_env.action_spec(),
        actor_net=actor_critic_net, value_net=actor_critic_net, optimizer=optimizer, num_epochs=10,
    )
    agent.initialize()

    train_step_counter = tf.Variable(0, name="train_step_counter")
    checkpoint = tf.train.Checkpoint(
        actor_critic_net=actor_critic_net, optimizer=optimizer, train_step_counter=train_step_counter
    )
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

    # --- モデルのロード戦略 ---
    if checkpoint_manager.latest_checkpoint:
        print(f"Restoring from checkpoint: {checkpoint_manager.latest_checkpoint}")
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    elif os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading weights from pre-trained model: {PRETRAINED_MODEL_PATH}")
        # Kerasモデルの重みをロード
        actor_critic_net.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
    else:
        print("No checkpoint or pre-trained model found. Starting from scratch.")


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
        # 1局分のデータを収集
        final_time_step, _ = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env, agent.collect_policy, observers=[replay_buffer.add_batch], num_episodes=COLLECT_EPISODES_PER_ITERATION
        ).run(time_step)

        # 収集したデータでモデルを学習
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        replay_buffer.clear()
        time_step = final_time_step # 次のループの開始状態を設定
        step = agent.train_step_counter.numpy()
        
        # 局の結果をサマリーに追加
        last_episode_yaku = py_env.get_info().get('last_yaku_result', [])
        if last_episode_yaku:
            yaku_counter.update(last_episode_yaku)
        
        # 報酬を記録
        reward_value = final_time_step.reward.numpy()[0]
        total_rewards.append(reward_value)

        # ログとダッシュボードの更新
        if i > 0 and i % LOG_INTERVAL == 0:
            avg_reward = np.mean(total_rewards)
            summary_data = {
                "step": int(step),
                "loss": float(train_loss.loss.numpy()),
                "avg_reward": float(avg_reward),
                "yaku_summary": yaku_counter.most_common(10)
            }
            await broadcast(json.dumps(summary_data))
            print(f"Step: {step}, Avg Reward (last {len(total_rewards)} games): {avg_reward:.2f}, Loss: {summary_data['loss']:.4f}")
            total_rewards = []

        # 牌譜の保存
        if step > 0 and step % KIFU_SAVE_INTERVAL == 0:
            kifu_path = os.path.join(KIFU_DIR, f"kifu_step_{step}.json")
            with open(kifu_path, 'w', encoding='utf-8') as f:
                json.dump(py_env.get_info()['game_events'], f, ensure_ascii=False, indent=2)
            print(f"🀄 Kifu saved to {kifu_path}")

        # チェックポイントの保存
        if step > 0 and step % 1000 == 0:
            checkpoint_manager.save()
            print(f"💾 Checkpoint saved at step {step}")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.create_task(start_server())
        loop.run_until_complete(main_training_loop())
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        ngrok.kill()
        loop.close()

