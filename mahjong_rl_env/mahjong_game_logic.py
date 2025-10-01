# -*- coding: utf-8 -*-
"""
TF-Agentsから独立した、純粋な麻雀のゲーム進行ロジック。
状態管理、ルール適用、報酬計算などを担当する。
"""
import numpy as np
import random
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

# プロジェクトルートからモジュールをインポート
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vectorizer import (
    TILE_34_COUNT, TOTAL_ACTION_DIM,
    ACTION_DISCARD_OFFSET, ACTION_RIICHI, ACTION_AGARI_TSUMO
)

class MahjongGameLogic:
    def __init__(self, rl_agent_id=0):
        self.rl_agent_id = rl_agent_id
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()
        self.rule_config = HandConfig(options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))
        self.reset()

    def reset(self):
        """局の状態を初期化"""
        self.game_state = {
            'events': [], 'scores': [25000] * 4, 'turn_num': 0, 'riichi_sticks': 0,
            'dora_indicators': [random.randint(0, 135)],
            'hands_136': [[] for _ in range(4)], 'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)], 'is_riichi': [False] * 4,
            'last_drawn_tile': [None] * 4,
            'total_tiles': list(range(136)), 'remaining_tiles': 70,
            'oya_player_id': random.randint(0, 3)
        }
        self.is_terminated = False
        self._deal_tiles()
        self.current_player_id = self.game_state['oya_player_id']
        self._draw_tile(self.current_player_id)
        self._add_event('INIT', {})
        return self.get_state()

    def get_state(self):
        return self.game_state

    def execute_rl_agent_action(self, action_id):
        action_id = int(action_id)
        player_id = self.rl_agent_id
        reward = 0.0

        if ACTION_DISCARD_OFFSET <= action_id < ACTION_RIICHI:
            tile_34 = action_id - ACTION_DISCARD_OFFSET
            self._execute_discard(player_id, tile_34)
            # TODO: 他プレイヤーのロン判定
        elif action_id == ACTION_RIICHI:
            # TODO: リーチ宣言の処理
            self._add_event('RIICHI', {'player': player_id})
            # リーチ後、ツモ牌をそのまま打牌する簡易ロジック
            drawn_tile_34 = TilesConverter.to_34_array([self.game_state['last_drawn_tile'][player_id]])
            discard_tile_idx = np.where(drawn_tile_34 == 1)[0][0]
            self._execute_discard(player_id, discard_tile_idx)
        elif action_id == ACTION_AGARI_TSUMO:
            reward = self._execute_agari(player_id, is_tsumo=True)
            self.is_terminated = True

        self.current_player_id = (player_id + 1) % 4
        return reward

    def simulate_bot_turns(self):
        while self.current_player_id != self.rl_agent_id and not self.is_terminated:
            player_id = self.current_player_id
            self._draw_tile(player_id)
            if self.is_terminated: break
            mask = self.get_action_mask_for_player(player_id)
            legal_actions = np.where(mask == 1)[0]
            action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
            # Botのアクション実行 (簡易版)
            if action < TILE_34_COUNT: self._execute_discard(player_id, action)
            self.current_player_id = (player_id + 1) % 4

    def get_action_mask(self):
        return self.get_action_mask_for_player(self.rl_agent_id)

    def get_action_mask_for_player(self, player_id):
        mask = np.zeros(TOTAL_ACTION_DIM, dtype=np.int8)
        hand_136 = self.game_state['hands_136'][player_id]
        hand_34_array = TilesConverter.to_34_array(hand_136)

        # 1. 打牌アクション
        for tile_34, count in enumerate(hand_34_array):
            if count > 0: mask[ACTION_DISCARD_OFFSET + tile_34] = 1

        # 2. ツモ和了アクション
        drawn_tile = self.game_state['last_drawn_tile'][player_id]
        if drawn_tile is not None and self._can_agari(player_id, drawn_tile, is_tsumo=True):
            mask[ACTION_AGARI_TSUMO] = 1

        # 3. リーチアクション
        shanten = self.shanten_calculator.calculate_shanten(hand_34_array)
        if shanten == 0 and not self.game_state['is_riichi'][player_id]:
            mask[ACTION_RIICHI] = 1

        return mask

    def _execute_discard(self, player_id, tile_34):
        hand = self.game_state['hands_136'][player_id]
        tile_to_remove = -1
        # 手牌から該当する牌を1枚探して削除 (赤ドラなどを考慮しない簡易版)
        for tile_136 in hand:
            if tile_136 // 4 == tile_34:
                tile_to_remove = tile_136
                break
        if tile_to_remove != -1:
            hand.remove(tile_to_remove)
            self.game_state['rivers'][player_id].append(tile_to_remove)
            self._add_event('DISCARD', {'player': player_id, 'tile': tile_to_remove})

    def _execute_agari(self, player_id, is_tsumo):
        win_tile = self.game_state['last_drawn_tile'][player_id] if is_tsumo else self.game_state['last_discarded_tile']
        if win_tile is None: return 0

        result = self._calculate_hand_value(player_id, win_tile, is_tsumo)
        if result is None or result.error: return 0

        # 点数移動をスコアに反映し、報酬を計算
        # 簡易的に、支払い総額を報酬とする
        total_payment = result.cost['main'] + result.cost['additional']
        # TODO: 親子関係を考慮した詳細な点数移動
        self.game_state['scores'][player_id] += total_payment
        self._add_event('AGARI', {'player': player_id, 'score': total_payment})
        return float(total_payment)

    def _can_agari(self, player_id, win_tile, is_tsumo):
        result = self._calculate_hand_value(player_id, win_tile, is_tsumo)
        return result is not None and result.error is None

    def _calculate_hand_value(self, player_id, win_tile, is_tsumo):
        try:
            return self.hand_calculator.estimate_hand_value(
                tiles=self.game_state['hands_136'][player_id],
                win_tile=win_tile,
                melds=self.game_state['melds'][player_id],
                dora_indicators=self.game_state['dora_indicators'],
                config=self._get_hand_config(player_id, is_tsumo)
            )
        except Exception:
            return None

    def _get_hand_config(self, player_id, is_tsumo):
        player_wind = Meld.EAST + ((player_id - self.game_state['oya_player_id'] + 4) % 4)
        round_wind = Meld.EAST + (self.game_state.get('round', 0) // 4)
        return HandConfig(is_tsumo=is_tsumo, is_riichi=self.game_state['is_riichi'][player_id],
                          player_wind=player_wind, round_wind=round_wind, options=self.rule_config.options)

    def _add_event(self, event_id, data):
        event = {'event_id': event_id, 'turn': self.game_state['turn_num']}
        event.update(data)
        self.game_state['events'].append(event)

    def _deal_tiles(self):
        random.shuffle(self.game_state['total_tiles'])
        for i in range(4):
            hand = sorted(self.game_state['total_tiles'][:13])
            self.game_state['total_tiles'] = self.game_state['total_tiles'][13:]
            self.game_state['hands_136'][i] = hand
        self.game_state['remaining_tiles'] -= 52

    def _draw_tile(self, player_id):
        if self.game_state['remaining_tiles'] <= 14 or not self.game_state['total_tiles']:
            self.is_terminated = True # 流局
            return
        tile = self.game_state['total_tiles'].pop(0)
        self.game_state['hands_136'][player_id].append(tile)
        self.game_state['last_drawn_tile'][player_id] = tile
        self.game_state['remaining_tiles'] -= 1
        if player_id == self.game_state['oya_player_id']:
             self.game_state['turn_num'] += 1
        self._add_event('DRAW', {'player': player_id, 'tile': tile})

