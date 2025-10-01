# -*- coding: utf-8 -*-
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# プロジェクトルートからモジュールをインポート
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorizer import (
    get_context_vector, TOTAL_ACTION_DIM,
    CONTEXT_MAX_LEN, EVENT_VECTOR_DIM
)
from mahjong_rl_env.mahjong_game_logic import MahjongGameLogic # ゲーム進行ロジックを分離

class MahjongPyEnvironment(py_environment.PyEnvironment):
    """TF-Agentsと互換性のあるPythonベースの麻雀環境"""

    def __init__(self):
        super().__init__()
        self._game = MahjongGameLogic() # 実際の麻雀ロジックを持つインスタンス
        self.rl_agent_id = self._game.rl_agent_id

        # --- 1. 行動の仕様 (Action Spec) を定義 ---
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=TOTAL_ACTION_DIM - 1, name='action'
        )

        # --- 2. 観測の仕様 (Observation Spec) を定義 ---
        self._observation_spec = {
            'context': array_spec.ArraySpec(
                shape=(CONTEXT_MAX_LEN, EVENT_VECTOR_DIM),
                dtype=np.float32,
                name='context'
            ),
            'action_mask': array_spec.ArraySpec(
                shape=(TOTAL_ACTION_DIM,),
                dtype=np.int8,
                name='action_mask'
            )
        }

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """新しい局を開始"""
        self._game.reset()
        self._game.simulate_bot_turns() # RLエージェントの番までBotターンを進める
        obs = self._get_observation()
        return ts.restart(obs)

    def _step(self, action):
        """AIの行動を受けてゲームを1ステップ進める"""
        if self._game.is_terminated:
            return self.reset()

        reward = self._game.execute_rl_agent_action(action)
        self._game.simulate_bot_turns()
        obs = self._get_observation()

        if self._game.is_terminated:
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward=reward, discount=1.0)

    def _get_observation(self):
        """現在のゲーム状態から観測データを生成"""
        game_state = self._game.get_state()
        context = get_context_vector(game_state['events'], self.rl_agent_id)
        mask = self._game.get_action_mask()
        return {'context': context, 'action_mask': mask}

