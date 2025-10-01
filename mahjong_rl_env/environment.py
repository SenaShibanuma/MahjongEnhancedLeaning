# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import collections
import torch as th # Botの行動決定に必要
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld 
from .feature_converter import get_context_vector, convert_tile_136_to_34 # 特徴量変換ロジック

# ----------------------------------------------------------------------
# 定数定義 (feature_converterと同期)
# ----------------------------------------------------------------------
TILE_34_COUNT = 34 
TILE_136_COUNT = 136
TOTAL_ACTION_DIM = TILE_34_COUNT + 10 
CONTEXT_MAX_LEN = 50 
EVENT_VECTOR_DIM = 256

class MahjongEnv(gym.Env):
    """
    4プレイヤー全てが、学習中の同一ポリシーを使用して行動する自己対戦（Self-Play）環境。
    """
    metadata = {"render_modes": ["human"], "render_fps": 4} 

    def __init__(self, agent_id=0):
        super(MahjongEnv, self).__init__()
        
        self.rl_agent_id = agent_id 
        self.all_policies = [None] * 4 # 外部から注入されるポリシー
        
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()
        
        # 状態空間の定義
        self.observation_space = spaces.Dict({
            "context": spaces.Box(low=-np.inf, high=np.inf, 
                                  shape=(CONTEXT_MAX_LEN, EVENT_VECTOR_DIM), 
                                  dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, 
                                      shape=(TOTAL_ACTION_DIM,), 
                                      dtype=np.int8)
        })
        self.action_space = spaces.Discrete(TOTAL_ACTION_DIM)
        
        self.game_state = {}
        self.current_player = 0 
        self.is_terminated = False
        self.rule_config = HandConfig(options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))

    def set_policies(self, policies):
        """外部から全プレイヤーのポリシーを設定"""
        if len(policies) != 4:
            raise ValueError("Must provide policies for all 4 players.")
        self.all_policies = policies

    def reset(self, seed=None, options=None):
        """局を初期化し、RLエージェントの番が来るまでBotターンをシミュレートする"""
        super().reset(seed=seed)
        
        self._initialize_new_round() 
        self._deal_tiles()
        
        self.current_player = self.game_state['oya_player_id']
        # シャンテン数を計算して状態に格納 (修正: 副露を考慮)
        self.game_state['shanten'] = [self._calculate_shanten_for_player(i) for i in range(4)]
        
        self._draw_tile(self.current_player)

        # RLエージェントの番でなければ、Botターンをシミュレート
        if self.current_player != self.rl_agent_id:
            self._simulate_until_rl_turn()

        observation = self._get_observation(self.current_player)
        info = self.game_state
        return observation, info

    def step(self, action_id):
        """
        RLエージェントの行動を実行し、次のRLエージェントの番までBotターンをシミュレートする
        """
        if self.is_terminated:
            return self._get_observation(self.current_player), 0.0, True, False, self.game_state

        player = self.current_player # 常に self.rl_agent_id であるべき

        # 1. 行動の合法性チェック
        action_mask = self._get_action_mask(player)
        if action_mask[action_id] == 0:
            print(f"Warning: Illegal action {action_id} by RL Agent {player}. Punishing with -1000 reward.")
            return self._get_observation(player), -1000.0, False, False, self.game_state

        tile_or_type, action_type = self._decode_action(action_id)
        reward = 0.0
        
        # 2. RLエージェントの行動実行
        if action_type == 'DISCARD':
            # decode_actionでNoneが返る可能性を考慮
            if tile_or_type is None:
                print(f"Error: Decoded tile is None for discard action {action_id}.")
                return self._get_observation(player), -1000.0, False, False, self.game_state
                
            self._execute_discard(player, tile_or_type) 
            self._add_event('DISCARD', {'player': player, 'tile': tile_or_type, 'is_tedashi': 0})
            
            # 他家からの応答チェック (修正: 応答があれば current_player が変わる)
            found_response = self._check_opponent_response(player, tile_or_type)
            
            if not found_response:
                self.current_player = (player + 1) % 4
                
        elif action_type == 'TSUMO_AGARI':
            reward, terminated = self._execute_tsumo_agari(player)
            self._handle_termination(terminated)
            
        elif action_type == 'RIICHI':
            self._execute_riichi(player)
            discard_tile = self.game_state['last_drawn_tile'][player]
            self._execute_discard(player, discard_tile)
            self._add_event('DISCARD', {'player': player, 'tile': discard_tile, 'is_tedashi': 0})
            
            # 他家からの応答チェック (修正: 応答があれば current_player が変わる)
            found_response = self._check_opponent_response(player, discard_tile)
            
            if not found_response:
                self.current_player = (player + 1) % 4
        
        # TODO: その他のアクション PUNG, CHII, DAIMINKAN, ANKAN, KAKAN, RON_AGARI のロジック

        # 3. 局の終了チェックとBotターンシミュレーション
        if self.is_terminated:
            return self._get_observation(self.current_player), reward, True, False, self.game_state

        # RLエージェントの番が来るまでBotターンをシミュレート
        self._simulate_until_rl_turn()
        
        # RLエージェントのターンになったらツモ
        if not self.is_terminated and self.current_player == self.rl_agent_id:
            if not self._draw_tile(self.current_player):
                # ツモ失敗=流局
                self._handle_termination(False)
                return self._get_observation(self.current_player), reward, True, False, self.game_state
                
        observation = self._get_observation(self.current_player)
        truncated = False 
        info = self.game_state

        return observation, reward, self.is_terminated, truncated, info

    # ----------------------------------------------------------------------
    # 内部ロジック (Bot AIの自動処理 - 同一ポリシーを使用)
    # ----------------------------------------------------------------------

    def _calculate_shanten_for_player(self, player_id):
        """
        指定されたプレイヤーの現在のシャンテン数を計算する (修正: 副露面子を考慮)
        """
        hand_136 = self.game_state['hands_136'][player_id]
        melds = self.game_state['melds'][player_id]
        
        # mahjongライブラリのShanten計算は、手牌と副露面子の両方を受け取る
        # ただし、TilesConverter.to_34_arrayは手牌のみを想定
        hand_34 = TilesConverter.to_34_array(hand_136)
        
        # meldsがMeldオブジェクトのリストであることを想定
        # mahjongライブラリのShanten()はMeldオブジェクトを直接受け取る
        return self.shanten_calculator.calculate_shanten(hand_34, melds=melds)
    
    def _get_action_from_policy(self, player_id):
        """ポリシーから行動IDを取得する (PyTorch依存)"""
        policy = self.all_policies[player_id]
        
        if policy is None:
            return self._get_random_legal_action(player_id)
        
        obs = self._get_observation(player_id)
        
        # PyTorch Tensorに変換 (Batch Size=1)
        context_tensor = th.from_numpy(obs['context']).unsqueeze(0)
        mask_tensor = th.from_numpy(obs['action_mask']).unsqueeze(0)
        
        obs_tensor = {
            "context": context_tensor, 
            "action_mask": mask_tensor
        }
        
        # ポリシーから行動を予測
        with th.no_grad():
            action_id, _ = policy.predict(obs_tensor, deterministic=False) 
            
        return action_id.item()

    def _get_random_legal_action(self, player_id):
        """デバッグ/ポリシー未設定時のための、ランダムな合法行動を返す"""
        mask = self._get_action_mask(player_id)
        legal_actions = np.where(mask == 1)[0]
        if legal_actions.size > 0:
            return random.choice(legal_actions)
        return 34 # PASS (安全策)

    def _run_bot_turn(self, player):
        """単一のBot（非RLエージェント）のツモ、行動決定、実行を処理する"""
        
        if self.is_terminated:
            return
            
        if not self._draw_tile(player):
            # 牌山切れ
            self._handle_termination(False)
            return
            
        # シャンテン数を更新
        self.game_state['shanten'][player] = self._calculate_shanten_for_player(player)

        # 1. 行動選択 (学習中の同一ポリシーを使用)
        action_id = self._get_action_from_policy(player)
        
        # 2. 行動実行
        tile_or_type, action_type = self._decode_action(action_id)
        
        found_response = False
        
        if action_type == 'TSUMO_AGARI':
            _, terminated = self._execute_tsumo_agari(player)
            if terminated:
                self._handle_termination(True)
                return
                
        elif action_type == 'DISCARD' or action_type == 'PASS':
            # 打牌処理
            discard_tile = tile_or_type if action_type == 'DISCARD' else self.game_state['last_drawn_tile'][player]
            
            if discard_tile is None:
                 print(f"Error: Bot P{player} tried to discard None tile with action {action_id}. Falling back to PASS.")
                 self.current_player = (player + 1) % 4
                 return
                 
            self._execute_discard(player, discard_tile)
            self._add_event('DISCARD', {'player': player, 'tile': discard_tile, 'is_tedashi': 0})
            
            # 他家AIからの応答チェック (鳴き/ロン)
            found_response = self._check_opponent_response(player, discard_tile)
            
            if not found_response:
                self.current_player = (player + 1) % 4
                
        elif action_type == 'RIICHI':
            self._execute_riichi(player)
            discard_tile = self.game_state['last_drawn_tile'][player]
            
            if discard_tile is None:
                print(f"Error: Bot P{player} Riichi failed due to missing drawn tile. Falling back to next turn.")
                self.current_player = (player + 1) % 4
                return
                
            self._execute_discard(player, discard_tile)
            self._add_event('DISCARD', {'player': player, 'tile': discard_tile, 'is_tedashi': 0})
            
            found_response = self._check_opponent_response(player, discard_tile)
            
            if not found_response:
                self.current_player = (player + 1) % 4
                
        elif action_type in ['PUNG', 'CHII', 'DAIMINKAN', 'ANKAN', 'KAKAN']:
             # TODO: 鳴き/カンの複雑なロジックを実装
             self.current_player = (player + 1) % 4 # 暫定的にターンを進める
        else:
            self.current_player = (player + 1) % 4
                 
    def _simulate_until_rl_turn(self):
        """現在のプレイヤーがRLエージェントになるか、局が終了するまで、Botの行動を連続で処理する"""
        player = self.current_player
        while player != self.rl_agent_id and not self.is_terminated:
            self._run_bot_turn(player)
            player = self.current_player

    def _check_opponent_response(self, discarder, discarded_tile):
        """
        打牌に対する他家（Bot含む）の応答をチェックする (修正: ロン、ポン/大明槓の優先度をチェック)
        """
        
        # 1. ロン判定 (最優先: 全員に対してチェック)
        ron_candidates = []
        for i in range(1, 4):
            player = (discarder + i) % 4
            if self._can_agari_at_this_moment(player, discarded_tile, is_tsumo=False):
                # ロンが合法
                ron_candidates.append(player)
                
        if ron_candidates:
            # 複数ロン（ダブロン、トリプルロン）を許可し、全員に対して報酬計算と局終了処理を行う
            for player in ron_candidates:
                 # 簡略化: ロンが合法なら強制的にロンを選択させる (ポリシー判断なし)
                _, terminated = self._execute_ron_agari(player, discarded_tile)
                # 局が終了しているはずだが、念のため局終了フラグを立てる
                if terminated:
                    self._handle_termination(True)
            return True # ロンが発生したため、局が終了し、他の応答チェックは不要
        
        # 2. ポン/大明槓判定 (ロンがなければチェック)
        # 鳴きは、ロンが発生しない場合にのみ検討される
        
        # for i in [1, 2, 3]: # 順位は関係なく、ポン/カンが優先
        #     player = (discarder + i) % 4
            
        #     if self._can_pung_kan_at_this_moment(player, discarded_tile):
        #         # TODO: ここでBotポリシーからポン/カンを選択するかを判断し、選択されたら実行する
        #         pass
            
        # 3. チー判定 (下家のみチェック)
        # TODO: チーロジック
            
        return False

    # ----------------------------------------------------------------------
    # ユーティリティメソッド (省略なしで実装を継続)
    # ----------------------------------------------------------------------

    def _can_pung_kan_at_this_moment(self, player, discarded_tile):
        """ポン/大明槓が可能かどうかの簡易チェック"""
        hand_136 = self.game_state['hands_136'][player]
        hand_34 = TilesConverter.to_34_array(hand_136)
        tile_34 = discarded_tile // 4
        
        if hand_34[tile_34] >= 2:
            return True
        return False

    def _initialize_new_round(self):
        """Parserのreset_round_stateに相当"""
        self.game_state = {
            'events': [], 'round': 0, 'honba': 0, 
            'dora_indicators': [random.randint(0, TILE_136_COUNT-1)],
            'scores': [25000] * 4,
            'hands_136': [[] for _ in range(4)],
            'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)],
            'is_riichi': [False] * 4,
            'riichi_turn': [-1] * 4,
            'oya_player_id': random.randint(0, 3), 
            'remaining_tiles': 70, 
            'turn_num': 0, 
            'last_drawn_tile': [None] * 4,
            'total_tiles': [0] * TILE_136_COUNT, 
            'shanten': [9] * 4, # 初期シャンテン数
            'rules': {'has_kuitan': True, 'has_aka_dora': True}
        }
        self.is_terminated = False
        self._add_event('INIT', {'round': self.game_state['round'], 'honba': self.game_state['honba'], 'scores': self.game_state['scores']})

    def _deal_tiles(self):
        """牌山を作成し、配牌を行う"""
        self.game_state['total_tiles'] = list(range(TILE_136_COUNT))
        random.shuffle(self.game_state['total_tiles'])
        
        for i in range(4):
            hand = sorted(self.game_state['total_tiles'][:13])
            self.game_state['total_tiles'] = self.game_state['total_tiles'][13:]
            self.game_state['hands_136'][i] = hand
            
        self.game_state['remaining_tiles'] -= (13 * 4 + 14) 

    def _draw_tile(self, player):
        """ツモ処理"""
        if not self.game_state['total_tiles']:
            return False 
            
        drawn_tile = self.game_state['total_tiles'].pop(0)
        self.game_state['hands_136'][player].append(drawn_tile)
        self.game_state['hands_136'][player].sort()
        self.game_state['last_drawn_tile'][player] = drawn_tile
        self.game_state['remaining_tiles'] -= 1
        self.game_state['turn_num'] += 1
        self._add_event('DRAW', {'player': player, 'tile': drawn_tile})
        return True

    def _execute_discard(self, player, tile):
        """打牌処理 (手牌から牌を削除し、河に追加)"""
        # 手出し/ツモ切りの判断
        if tile in self.game_state['hands_136'][player]:
            self.game_state['hands_136'][player].remove(tile)
        else:
             if self.game_state['last_drawn_tile'][player] is not None and self.game_state['last_drawn_tile'][player] == tile:
                 # ツモ切りとして処理（last_drawn_tileは手牌に含まれているとみなす）
                 self.game_state['hands_136'][player].remove(tile)
                 self.game_state['last_drawn_tile'][player] = None
             else:
                 # ここに到達した場合、不正な打牌（手牌にもツモ牌にもない）
                 print(f"Error: Player {player} attempted to discard tile {tile} which is not in hand/drawn tile.")
                 
        self.game_state['rivers'][player].append(tile)
        self.game_state['last_discarded_tile'] = tile
        self.game_state['last_discarder'] = player

    def _execute_tsumo_agari(self, player):
        """ツモ和了処理と報酬計算"""
        win_tile = self.game_state['last_drawn_tile'][player]
        if win_tile is None:
            print(f"Error: P{player} attempted Tsumo Agari but last_drawn_tile is None.")
            return 0.0, False

        result = self._calculate_hand_value(player, win_tile, is_tsumo=True)
        
        if result and result.error is None:
            reward = self._get_tsumo_reward(result.cost, player)
            self._add_event('AGARI', {'winner': player, 'from': player, 'cost': result.cost})
            return reward, True
        else:
            return 0.0, False

    def _execute_ron_agari(self, player, winning_tile):
        """ロン和了処理と報酬計算"""
        result = self._calculate_hand_value(player, winning_tile, is_tsumo=False)
        
        if result and result.error is None:
            discarder = self.game_state.get('last_discarder', (player - 1) % 4)
            reward = self._get_ron_reward(result.cost, player, discarder)
            self._add_event('AGARI', {'winner': player, 'from': discarder, 'cost': result.cost})
            return reward, True
        else:
            return 0.0, False
            
    def _execute_riichi(self, player):
        """リーチ宣言の内部処理"""
        self.game_state['is_riichi'][player] = True
        self.game_state['riichi_turn'][player] = self.game_state['turn_num']
        self._add_event('RIICHI_DECLARED', {'player': player})

    def _calculate_hand_value(self, player, win_tile, is_tsumo):
        """HandCalculatorを呼び出し、点数結果オブジェクトを返す"""
        hand_136 = self.game_state['hands_136'][player]
        
        # 風の計算
        player_wind_id = (player - self.game_state['oya_player_id'] + 4) % 4
        round_wind_id = self.game_state['round'] // 4
        
        config = HandConfig(
            is_tsumo=is_tsumo,
            player_wind=Meld.EAST + player_wind_id, 
            round_wind=Meld.EAST + round_wind_id,
            is_riichi=self.game_state['is_riichi'][player],
            options=self.rule_config.options
        )
        
        return self.hand_calculator.estimate_hand_value(
            tiles=hand_136,
            win_tile=win_tile,
            melds=self.game_state['melds'][player],
            dora_indicators=self.game_state['dora_indicators'],
            config=config
        )

    def _get_tsumo_reward(self, cost, winner_id):
        """ツモ和了時の報酬計算 (獲得点数)"""
        oya = self.game_state['oya_player_id']
        if winner_id == oya: 
             payment_sum = cost['main'] * 3 + cost['additional'] * 3 + self.game_state['honba'] * 300 * 3
        else:
             payment_sum = cost['main'] + cost['additional'] * 2 + self.game_state['honba'] * 300
        return payment_sum

    def _get_ron_reward(self, cost, winner_id, loser_id):
        """ロン和了時の報酬計算 (獲得点数)"""
        main_score = cost['main'] * 2 + cost['additional']
        payment = main_score + self.game_state['honba'] * 300
        return payment

    def _handle_termination(self, forced_termination):
        """局の終了処理"""
        self.is_terminated = True
        if self.game_state['remaining_tiles'] <= 0 and not forced_termination:
            self._add_event('RYUKYOKU', {'reason': 'Haishan Kire'})
            
    def _add_event(self, event_id, data):
        """ゲームイベントを記録"""
        event = {'event_id': event_id, 'turn': self.game_state['turn_num']}
        event.update(data)
        self.game_state['events'].append(event)
