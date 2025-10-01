# -*- coding: utf-8 -*-
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
import os

from mahjong_rl_env.tf_agents_environment import MahjongPyEnvironment
from models.tf_agents_policy import MahjongActorCriticNetwork

# --- ハイパーパラメータ ---
NUM_ITERATIONS = 200_000
COLLECT_EPISODES_PER_ITERATION = 20
REPLAY_BUFFER_CAPACITY = 4000
LEARNING_RATE = 3e-5
LOG_INTERVAL = 200
EVAL_INTERVAL = 1000
CHECKPOINT_DIR = './rl_checkpoints'
PRETRAINED_MODEL_PATH = './models/pretrained_transformer.keras' # 事前学習済みモデルのパス

def train_agent():
    # --- 1. 環境の準備 ---
    py_env = MahjongPyEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    print("✅ TF-Agents Environment Ready.")

    # --- 2. ネットワークとエージェントの準備 ---
    actor_critic_net = MahjongActorCriticNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec()
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    agent = ppo_agent.PPOAgent(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        actor_net=actor_critic_net,
        value_net=actor_critic_net,
        optimizer=optimizer,
        num_epochs=10,
    )
    agent.initialize()
    print("✅ PPO Agent with Transformer Network Ready.")

    # --- 3. チェックポイントと事前学習済みモデルのロード ---
    train_step_counter = tf.Variable(0)
    checkpoint = tf.train.Checkpoint(
        actor_critic_net=actor_critic_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, CHECKPOINT_DIR, max_to_keep=5
    )

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"✅ Checkpoint restored from {checkpoint_manager.latest_checkpoint}")
    elif os.path.exists(PRETRAINED_MODEL_PATH):
        # チェックポイントがない場合のみ、事前学習済みモデルを試す
        actor_critic_net.load_weights(PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)
        print(f"✅ Pre-trained weights loaded from {PRETRAINED_MODEL_PATH}")

    # --- 4. データ収集と学習ループ ---
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=REPLAY_BUFFER_CAPACITY
    )
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=COLLECT_EPISODES_PER_ITERATION
    )

    print("\n--- Starting Reinforcement Learning Training ---")
    agent.train = common.function(agent.train)

    for i in range(NUM_ITERATIONS):
        driver.run()
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        replay_buffer.clear()
        
        step = agent.train_step_counter.numpy()

        if step % LOG_INTERVAL == 0:
            print(f"Step: {step}, Loss: {train_loss.loss.numpy()}")
        
        if step % EVAL_INTERVAL == 0:
            # TODO: ここに評価ロジックを追加
            pass
        
        if step % 1000 == 0: # 1000ステップごとに保存
            checkpoint_manager.save()
            print(f"Checkpoint saved at step {step}")


if __name__ == '__main__':
    train_agent()

