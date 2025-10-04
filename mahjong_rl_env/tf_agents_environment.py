# -*- coding: utf-8 -*-
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# 循環参照を避けるため、必要なものだけをインポート
from vectorizer import (
    get_observation_vector,
    ACTION_SPACE,
    EVENT_VECTOR_DIM,
    CONTEXT_MAX_LEN
)
from mahjong_rl_env.mahjong_game_logic import MahjongGameLogic # ゲーム進行ロ_ジックを分離

class MahjongPyEnvironment(py_environment.PyEnvironment):
    """TF-Agentsと連携するためのPython環境ラッパー"""
    def __init__(self):
        super().__init__()
        self._game = MahjongGameLogic()

        # 1. 行動空間の定義 (AIの出力)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(ACTION_SPACE) - 1, name='action')

        # 2. 観測空間の定義 (AIへの入力)
        self._observation_spec = {
            'observation': array_spec.BoundedArraySpec(
                shape=(CONTEXT_MAX_LEN, EVENT_VECTOR_DIM), dtype=np.float32, name='observation'),
            # --- 修正箇所: データ型を int8 から int32 に変更 ---
            'action_mask': array_spec.BoundedArraySpec(
                shape=(len(ACTION_SPACE),), dtype=np.int32, name='action_mask')
        }
        self._state = self._game.reset()
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self._game.reset()
        self._episode_ended = False
        return ts.restart(self._get_observation())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # TF-Agentsから受け取った行動をゲームロジックに渡す
        player_id = self._state['current_player_id']
        next_state, reward, done, info = self._game.step(player_id, action)
        self._state = next_state

        if done:
            self._episode_ended = True
            # 局が終了した場合、報酬を伴う最終ステップを返す
            return ts.termination(self._get_observation(), reward)
        else:
            # 局が継続する場合、通常の遷移ステップを返す
            return ts.transition(self._get_observation(), reward=reward, discount=1.0)

    def _get_observation(self):
        """現在のゲーム状態から、AIへの観測データを生成する"""
        player_id = self._state['current_player_id']
        
        observation_vector = get_observation_vector(
            self._game.get_state_for_vectorizer(player_id)
        )
        action_mask = self._game.get_action_mask(player_id)
        
        return {
            'observation': observation_vector,
            'action_mask': action_mask,
        }

