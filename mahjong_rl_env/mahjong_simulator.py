# mahjong_simulator.py
# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld

# 既存の feature_converter.py からベクトル化ロジックをインポート
# このファイルはシミュレーターと同じディレクトリに配置してください。
from feature_converter import get_context_vector, convert_tile_136_to_34

# ----------------------------------------------------------------------
# 定数定義 (feature_converter.py と完全に一致させる)
# ----------------------------------------------------------------------
TILE_34_COUNT = 34
# 行動の総数: 打牌(34種) + チー(3種) + ポン + カン + リーチ + ツモ和了 + パス
# ご自身のモデルの出力次元数に合わせて調整してください。
TOTAL_ACTION_DIM = 44
CONTEXT_MAX_LEN = 50
EVENT_VECTOR_DIM = 256 # feature_converter.py で定義されたベクトル次元

class MahjongEnv(gym.Env):
    """
    天鳳ログでの事前学習モデルと連携する、自己対戦（Self-Play）用の強化学習環境。
    AIエージェントと、同一ポリシーで動く3体のBotで対戦が進行します。
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, rl_agent_id=0):
        super(MahjongEnv, self).__init__()

        self.rl_agent_id = rl_agent_id  # 強化学習エージェントのプレイヤーID (0-3)
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()

        # --- AIへの入力 (Observation Space) の定義 ---
        # この構造は、MaskedTransformerPolicyが期待する入力形式と一致しています。
        self.observation_space = spaces.Dict({
            # 状況ベクトル: (シーケンス長, ベクトル次元)
            "context": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(CONTEXT_MAX_LEN, EVENT_VECTOR_DIM),
                dtype=np.float32
            ),
            # 合法手マスク: (行動の総数,)
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(TOTAL_ACTION_DIM,),
                dtype=np.int8
            )
        })

        # --- AIからの出力 (Action Space) の定義 ---
        self.action_space = spaces.Discrete(TOTAL_ACTION_DIM)

        self.game_state = {}
        self.current_player_id = 0
        self.is_terminated = False
        # ルール設定（クイタンあり、赤ドラありなど）
        self.rule_config = HandConfig(options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))
        print("✅ 麻雀シミュレーターの初期化が完了しました。")


    def reset(self, seed=None, options=None):
        """新しい局を開始し、最初のエージェントの観測を返す"""
        super().reset(seed=seed)
        self._initialize_new_round() # 局の状態をリセット
        self._deal_tiles()           # 配牌

        self.current_player_id = self.game_state['oya_player_id']
        self._draw_tile(self.current_player_id) # 親が最初のツモ

        # もし開始プレイヤーがRLエージェントでない場合、エージェントの番までBotを自動で動かす
        if self.current_player_id != self.rl_agent_id:
            self._simulate_bot_turns()

        observation = self._get_observation_for_player(self.rl_agent_id)
        info = self._get_info()

        return observation, info

    def step(self, action_id):
        """
        エージェントからの行動を受け取り、ゲームを1ステップ進める
        """
        if self.is_terminated:
            # ゲーム終了後は、最後の観測とゼロ報酬を返す
            obs = self._get_observation_for_player(self.rl_agent_id)
            return obs, 0.0, True, False, self._get_info()

        # --- 1. RLエージェントの行動を実行 ---
        # action_idからゲーム内アクションへの変換と実行
        # (この部分は、実際のゲームロジックに合わせて詳細な実装が必要)
        # 例: self._execute_action(self.rl_agent_id, action_id)
        # (打牌、リーチ、和了などの処理)
        # ...

        # 仮に打牌アクションとして処理
        discard_tile_34 = action_id
        if discard_tile_34 < TILE_34_COUNT:
             # (手牌から牌を捨てるロジック)
            self._add_event('DISCARD', {'player': self.rl_agent_id, 'tile': discard_tile_34 * 4}) # 136牌に変換
            print(f"エージェント(P{self.rl_agent_id})が牌 {discard_tile_34} を捨てました。")
        else:
            # その他のアクション（リーチ、和了など）
            print(f"エージェント(P{self.rl_agent_id})がアクション {action_id} を実行しました。")


        # 報酬は一旦0で固定（実際の和了/放銃ロジックで計算）
        reward = 0.0

        # --- 2. 次のプレイヤーへターンを移行 ---
        self.current_player_id = (self.rl_agent_id + 1) % 4

        # --- 3. RLエージェントの次の番までBotターンを進行 ---
        if not self.is_terminated:
            self._simulate_bot_turns()

        # --- 4. 最終的な結果を返す ---
        observation = self._get_observation_for_player(self.rl_agent_id)
        info = self._get_info()
        truncated = False # ゲームが時間等で打ち切られたか

        # (局の終了条件をチェックし、is_terminatedを更新)
        # 例: if self.game_state['remaining_tiles'] == 0: self.is_terminated = True

        return observation, reward, self.is_terminated, truncated, info

    def _simulate_bot_turns(self):
        """現在のプレイヤーがRLエージェントになるまで、Botのターンを自動進行させる"""
        while self.current_player_id != self.rl_agent_id and not self.is_terminated:
            player = self.current_player_id
            print(f"--- Bot (P{player}) のターン ---")

            # Botの観測データを生成
            bot_obs = self._get_observation_for_player(player)

            # Botの行動選択 (方策はRLエージェントと同一と仮定)
            # ここでは、合法手の中からランダムに行動を選択する簡易的なAIを実装
            legal_actions = np.where(bot_obs["action_mask"] == 1)[0]
            action_id = np.random.choice(legal_actions) if len(legal_actions) > 0 else 34 # 打牌0萬

            # Botの行動実行 (簡易版)
            print(f"Bot (P{player}) が行動 {action_id} を選択しました。")
            # (self._execute_action(player, action_id) のような処理)
            # ...

            # ターンを進める
            self.current_player_id = (self.current_player_id + 1) % 4


    def _get_observation_for_player(self, player_id):
        """指定されたプレイヤー視点の観測（ContextとAction Mask）を生成"""
        # --- Contextベクトルの生成 ---
        # `feature_converter.get_context_vector` を呼び出す
        # この関数は、教師あり学習のデータ生成時と全く同じロジックである必要があります。
        context_vector = get_context_vector(self.game_state, player_id)

        # --- Action Maskの生成 ---
        # 現状で実行可能な行動（打牌、リーチなど）を判定し、1のマスクを作成する
        action_mask = self._get_action_mask(player_id)

        return {"context": context_vector, "action_mask": action_mask}

    def _get_action_mask(self, player_id):
        """プレイヤーが現在取れる合法な行動のマスクを返す"""
        mask = np.zeros(TOTAL_ACTION_DIM, dtype=np.int8)
        # (実装例)
        # 1. 打牌可能な牌をマスク=1にする
        #    hand_34 = TilesConverter.to_34_array(self.game_state['hands_136'][player_id])
        #    for tile_34, count in enumerate(hand_34):
        #        if count > 0: mask[tile_34] = 1
        # 2. リーチ可能なら、リーチアクションのインデックスをマスク=1にする
        # 3. 和了可能なら、和了アクションのインデックスをマスク=1にする
        # ...
        # (簡易的に、全ての打牌を可能としてマスクを作成)
        mask[:TILE_34_COUNT] = 1
        return mask

    # ======================================================================
    # 以下、補助的な内部関数 (初期化、イベント追加など)
    # ======================================================================

    def _initialize_new_round(self):
        """局の内部状態をリセット"""
        self.game_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0,
            'dora_indicators': [random.randint(0, 135)],
            'scores': [25000] * 4,
            'hands_136': [[] for _ in range(4)], 'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)], 'is_riichi': [False] * 4,
            'oya_player_id': random.randint(0, 3),
            'remaining_tiles': 70, 'turn_num': 0,
            'last_drawn_tile': [None] * 4,
            'total_tiles': list(range(136)),
        }
        self.is_terminated = False
        self._add_event('INIT', {'scores': self.game_state['scores']})

    def _deal_tiles(self):
        """牌を配る"""
        random.shuffle(self.game_state['total_tiles'])
        for i in range(4):
            hand = sorted(self.game_state['total_tiles'][:13])
            self.game_state['total_tiles'] = self.game_state['total_tiles'][13:]
            self.game_state['hands_136'][i] = hand
        self.game_state['remaining_tiles'] -= (13 * 4)

    def _draw_tile(self, player_id):
        """牌をツモる"""
        if not self.game_state['total_tiles']:
            self.is_terminated = True # 流局
            return
        drawn_tile = self.game_state['total_tiles'].pop(0)
        self.game_state['hands_136'][player_id].append(drawn_tile)
        self.game_state['hands_136'][player_id].sort()
        self.game_state['last_drawn_tile'][player_id] = drawn_tile
        self.game_state['remaining_tiles'] -= 1
        self.game_state['turn_num'] += 1
        self._add_event('DRAW', {'player': player_id, 'tile': drawn_tile})

    def _add_event(self, event_id, data):
        """ゲームイベントを履歴に追加"""
        event = {'event_id': event_id, 'turn': self.game_state['turn_num']}
        event.update(data)
        self.game_state['events'].append(event)
        # print(f"Event Added: {event}") # デバッグ用

    def _get_info(self):
        """デバッグや分析用の追加情報を返す"""
        return {
            "scores": self.game_state['scores'],
            "turn": self.game_state['turn_num'],
            "remaining_tiles": self.game_state['remaining_tiles']
        }

    def render(self, mode='human'):
        """現在のゲーム状態を描画する（デバッグ用）"""
        if mode == 'ansi':
            return self._render_text()
        # humanモードでは、より詳細なグラフィカル表示も可能
        print(self._render_text())

    def _render_text(self):
        """テキストベースで現在の盤面を出力"""
        output = f"\n===== Turn: {self.game_state['turn_num']} | Current Player: P{self.current_player_id} =====\n"
        output += f"Scores: {self.game_state['scores']}\n"
        output += f"Dora Indicator: {self.game_state['dora_indicators'][0]}\n"
        for i in range(4):
            player_str = f"P{i} (Agent)" if i == self.rl_agent_id else f"P{i} (Bot)"
            hand_str = " ".join([str(t) for t in self.game_state['hands_136'][i]])
            river_str = " ".join([str(t) for t in self.game_state['rivers'][i]])
            output += f"{player_str}:\n"
            output += f"  Hand: {hand_str}\n"
            output += f"  River: {river_str}\n"
        return output

# ======================================================================
# シミュレーターの動作テスト
# ======================================================================
if __name__ == '__main__':
    print("--- シミュレーターの動作テストを開始します ---")
    env = MahjongEnv(rl_agent_id=0)
    obs, info = env.reset()

    # 観測データの形式を確認
    print("\n[初回観測データ]")
    print(f"Context Shape: {obs['context'].shape}")
    print(f"Action Mask Shape: {obs['action_mask'].shape}")
    print(f"Legal Actions (Count): {np.sum(obs['action_mask'])}")

    env.render() # 初回盤面の表示

    # 10ターン分のシミュレーションを実行
    for i in range(10):
        if env.is_terminated:
            print("\n--- ゲームが終了しました ---")
            break

        # エージェントは、合法手の中からランダムに行動を選択
        action_mask = obs['action_mask']
        legal_actions = np.where(action_mask == 1)[0]
        random_action = np.random.choice(legal_actions)

        print(f"\n>>> エージェントがランダムな行動 {random_action} を選択...")

        # 環境を1ステップ進める
        obs, reward, terminated, truncated, info = env.step(random_action)

        env.render() # 各ステップ後の盤面を表示
        print(f"Reward: {reward}, Terminated: {terminated}, Info: {info}")

    env.close()