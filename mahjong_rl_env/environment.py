# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import collections
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld # Meld.decode_m などを使用

# ----------------------------------------------------------------------
# 定数定義
# ----------------------------------------------------------------------

# 牌の種類数 (萬子, 筒子, 索子, 字牌)
TILE_34_COUNT = 34 
# 136牌IDの総数 (0-135)
TILE_136_COUNT = 136
# 全ての行動の総数 (打牌34種 + 特殊行動10種 + 鳴きのバリエーション)
# 簡単化のため、特殊行動 + 打牌34種と定義。鳴きは後でエンコードする
TOTAL_ACTION_DIM = TILE_34_COUNT + 10 
CONTEXT_MAX_LEN = 50 
EVENT_VECTOR_DIM = 256 # 仮のイベントベクトル次元

class MahjongEnv(gym.Env):
    """
    強化学習用の麻雀環境 (Gymnasium互換)

    状態空間 (Observation Space):
    - context: 過去のイベントシーケンス (Transformerの入力)
    - action_mask: 合法的な行動を示すマスク

    行動空間 (Action Space):
    - 離散空間: 全ての可能なアクションID
    """
    metadata = {"render_modes": ["human"], "render_fps": 4} 

    def __init__(self, agent_id=0, opponent_policy=None):
        """
        :param agent_id: 強化学習を行うエージェントのID (0-3)
        :param opponent_policy: 他家AIの行動を決定する関数 (Noneの場合はランダム)
        """
        super(MahjongEnv, self).__init__()
        
        # RLを行うエージェントのIDを設定
        self.rl_agent_id = agent_id 
        # 他家の行動を決定するポリシー
        self.opponent_policy = opponent_policy
        
        # mahjongライブラリのコアエンジン
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()
        self.tiles_converter = TilesConverter()
        
        # 状態空間の定義 (あなたの仕様書に基づく)
        self.observation_space = spaces.Dict({
            # 文脈 (Context): 最大長50の256次元ベクトル
            "context": spaces.Box(low=-np.inf, high=np.inf, 
                                  shape=(CONTEXT_MAX_LEN, EVENT_VECTOR_DIM), 
                                  dtype=np.float32),
            # 行動マスク: 全ての行動IDに対応するブール値 (合法性)
            "action_mask": spaces.Box(low=0, high=1, 
                                      shape=(TOTAL_ACTION_DIM,), 
                                      dtype=np.int8)
        })
        
        # 行動空間の定義: 離散的な行動ID
        # 0-33: 打牌 (34種)
        # 34: ACTION_PASS
        # 35: ACTION_TSUMO_AGARI
        # 36: ACTION_RON_AGARI
        # 37: ACTION_RIICHI (宣言)
        # 38: ACTION_PUNG (ポン)
        # 39: ACTION_CHII (チー)
        # 40: ACTION_DAIMINKAN (大明槓)
        # 41: ACTION_ANKAN (暗槓)
        # 42: ACTION_KAKAN (加槓)
        # ... (合計 TOTAL_ACTION_DIM)
        self.action_space = spaces.Discrete(TOTAL_ACTION_DIM)
        
        # ゲーム内部状態 (TransformerParserのround_stateに相当)
        self.game_state = {}
        self.current_player = 0 # 現在行動中のプレイヤーID (0-3)
        self.is_terminated = False
        
        # ルール設定（天鳳デフォルトを想定）
        self.rule_config = HandConfig(options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))

    # ----------------------------------------------------------------------
    # Gymnasium 必須メソッド
    # ----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        局を初期化し、最初の観測を返す
        """
        super().reset(seed=seed)
        
        # 局の内部状態を初期化
        self._initialize_new_round() 
        
        # 最初の配牌とツモ
        self._deal_tiles()
        
        # 親の番からスタート
        self.current_player = self.game_state['oya_player_id']
        
        # RLエージェントの番まで、Botに対局を進めさせる
        self._process_opponent_turns(start_player=self.game_state['oya_player_id'])
        
        # RLエージェントのツモ処理
        self._draw_tile(self.current_player)

        observation = self._get_observation(self.current_player)
        info = self.game_state
        return observation, info

    def step(self, action_id):
        """
        RLエージェントの行動を実行し、報酬、次の状態、終了フラグを返す
        """
        if self.is_terminated:
            return self._get_observation(self.current_player), 0.0, True, False, self.game_state

        player = self.current_player # これは常に self.rl_agent_id であるべき

        # 1. 行動の合法性チェック
        action_mask = self._get_action_mask(player)
        if action_mask[action_id] == 0:
            print(f"Warning: Illegal action {action_id} taken by player {player}. Punishing with -1000 reward.")
            # 不正行動はペナルティ
            return self._get_observation(player), -1000.0, False, False, self.game_state

        # 2. 行動の実行と状態更新
        tile_or_type, action_type = self._decode_action(action_id)
        
        # 3. 報酬と終了判定の初期化
        reward = 0.0
        terminated = False
        
        # 行動ロジック実行 (RLエージェント)
        if action_type == 'DISCARD':
            self._execute_discard(player, tile_or_type) 
            self._add_event('DISCARD', {'player': player, 'tile': tile_or_type})
            
            # 他家からの応答（鳴き、ロン）をチェック (この実装では省略)
            # 応答がない場合、次のプレイヤーにターンを渡す
            self.current_player = (player + 1) % 4

        elif action_type == 'TSUMO_AGARI':
            reward, terminated = self._execute_tsumo_agari(player)
            
        elif action_type == 'PASS':
            # パス => ツモ切りと同じとして打牌処理
            self._execute_discard(player, self.game_state['last_drawn_tile'][player])
            self._add_event('DISCARD', {'player': player, 'tile': self.game_state['last_drawn_tile'][player]})
            self.current_player = (player + 1) % 4
            
        elif action_type == 'RIICHI':
            # リーチ宣言牌は、次のDISCARDタグで処理されるべきだが、ここでは簡単化のため、
            # ツモ切りとして処理し、リーチフラグを立てる
            self._execute_riichi(player)
            self._execute_discard(player, self.game_state['last_drawn_tile'][player])
            self._add_event('DISCARD', {'player': player, 'tile': self.game_state['last_drawn_tile'][player]})
            self.current_player = (player + 1) % 4
            
        elif action_type in ['PUNG', 'CHII', 'DAIMINKAN', 'ANKAN', 'KAKAN', 'RON_AGARI']:
             # TODO: 複雑な鳴き/カン/ロンアクションの実行ロジック
             # 例: ロン
             if action_type == 'RON_AGARI':
                 # ロン和了はstepの直前で他家から打牌があった場合に発生。
                 # ここでは、打牌された牌はgame_stateに保持されている前提で処理
                 winning_tile = self.game_state.get('last_discarded_tile', None)
                 if winning_tile is not None:
                     reward, terminated = self._execute_ron_agari(player, winning_tile)
                 else:
                     reward = -1000.0 # 不正なロン

        # 4. 局の終了チェック
        if terminated or self.game_state['remaining_tiles'] <= 0:
            reward, terminated = self._handle_termination(terminated)
            self.is_terminated = terminated
            
        # 5. RLエージェントの番が来るまでBotに対局を進めさせる
        if not terminated:
            # 次のプレイヤーにターンを渡した後、Botターン処理を開始
            self._process_opponent_turns(start_player=self.current_player)
            
            # RLエージェントのターンになったらツモ
            if not self.is_terminated and self.current_player == self.rl_agent_id:
                self._draw_tile(self.current_player)
                
        observation = self._get_observation(self.current_player)
        truncated = False 
        info = self.game_state

        return observation, reward, terminated, truncated, info

    # ----------------------------------------------------------------------
    # 内部ロジック (他家AIの自動処理)
    # ----------------------------------------------------------------------

    def _process_opponent_turns(self, start_player):
        """
        現在のプレイヤーがRLエージェントになるまで、他家の行動を処理する
        """
        player = start_player
        while player != self.rl_agent_id and not self.is_terminated:
            
            # 1. ツモ処理
            if not self._draw_tile(player):
                 # 牌山切れ
                 self._handle_termination(False)
                 return

            # 2. 行動選択 (Bot Policy)
            # ここでは簡単化のため、ランダムまたはPASS/ツモ切りを優先
            action_id = self._get_bot_action(player)
            
            # 3. 行動実行
            tile_or_type, action_type = self._decode_action(action_id)
            
            if action_type == 'TSUMO_AGARI':
                # ツモ和了したら終了
                _, terminated = self._execute_tsumo_agari(player)
                if terminated:
                    self._handle_termination(True)
                    return
            elif action_type in ['DISCARD', 'PASS']:
                # 打牌処理 (ツモ切り/手出し)
                discard_tile = tile_or_type if action_type == 'DISCARD' else self.game_state['last_drawn_tile'][player]
                self._execute_discard(player, discard_tile)
                self._add_event('DISCARD', {'player': player, 'tile': discard_tile})
                
                # 他家AIからの応答チェック (鳴き/ロン)
                # TODO: ここでロン/鳴きの応答ロジックを実装
                
                # 応答がなければ次のプレイヤーへ
                player = (player + 1) % 4
            elif action_type == 'RIICHI':
                # リーチ
                self._execute_riichi(player)
                discard_tile = self.game_state['last_drawn_tile'][player]
                self._execute_discard(player, discard_tile)
                self._add_event('DISCARD', {'player': player, 'tile': discard_tile})
                player = (player + 1) % 4
            else:
                 # その他の特殊アクションはここではランダム選択肢から除外すべき
                 player = (player + 1) % 4
                 
            self.current_player = player # ターンの更新

    def _get_bot_action(self, player):
        """
        他家AIの行動を決定する
        - opponent_policyが設定されていればそれを使用（セルフプレイ）
        - なければランダムまたは簡単なヒューリスティックを使用
        """
        action_mask = self._get_action_mask(player)
        
        # 1. 和了優先
        if action_mask[35] == 1: # TSUMO_AGARI
            return 35
        
        # 2. リーチ可能ならリーチ優先
        if action_mask[37] == 1:
            return 37

        # 3. それ以外は合法的な打牌からランダムに選択
        legal_discards = np.where(action_mask[:TILE_34_COUNT] == 1)[0]
        if legal_discards.size > 0:
            # ツモ切り（打牌した牌がツモ牌と同じになるように）
            drawn_tile_34 = self.game_state['last_drawn_tile'][player] // 4
            if drawn_tile_34 in legal_discards:
                return drawn_tile_34 # ツモ切り
            else:
                return random.choice(legal_discards) # 手出し
        
        # 牌がない場合などはPASS (理論上ありえない)
        return 34 # ACTION_PASS

    def _execute_riichi(self, player):
        """リーチ宣言の内部処理"""
        self.game_state['is_riichi'][player] = True
        self.game_state['riichi_turn'][player] = self.game_state['turn_num']
        # 供託棒の移動処理は報酬計算で実施される前提
        self._add_event({'event_id': 'RIICHI_DECLARED', 'player': player})

    def _check_opponent_response(self, discarder, discarded_tile):
        """
        打牌に対する他家（Bot）の応答（鳴き、ロン）をチェックする
        このロジックは非常に複雑になるため、セルフプレイAIで学習させる際に重要になる
        ここでは、ロンが可能な場合のみチェックするダミーロジックを保持
        """
        found_response = False
        
        for player in [(discarder + i) % 4 for i in [1, 2, 3]]: # 順番にチェック
            if player == self.rl_agent_id:
                # RLエージェントのロンは、stepの外側で処理されるべき
                continue 
            
            # ロン判定
            if self._can_agari_at_this_moment(player, discarded_tile, is_tsumo=False):
                # Botがロンを選択 (ここでは強制的にロンを選択させる)
                _, terminated = self._execute_ron_agari(player, discarded_tile)
                if terminated:
                    self._handle_termination(True)
                    return True # 局終了
                    
        return found_response

    # ----------------------------------------------------------------------
    # 以下、既存のロジックを保持 (変更なし)
    # ----------------------------------------------------------------------
    
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
            'oya_player_id': random.randint(0, 3), # 親をランダムに決定
            'remaining_tiles': 70, 
            'turn_num': 0, 
            'last_drawn_tile': [None] * 4,
            'total_tiles': [0] * TILE_136_COUNT # 136牌山
        }
        self.is_terminated = False
        self._add_event('INIT', {'round': self.game_state['round'], 'honba': self.game_state['honba']})

    def _deal_tiles(self):
        """牌山を作成し、配牌を行う"""
        # 1. 牌山作成 (136枚)
        self.game_state['total_tiles'] = list(range(TILE_136_COUNT))
        random.shuffle(self.game_state['total_tiles'])
        
        # 2. 配牌 (13枚ずつ)
        for i in range(4):
            # 13枚配る
            hand = sorted(self.game_state['total_tiles'][:13])
            self.game_state['total_tiles'] = self.game_state['total_tiles'][13:]
            self.game_state['hands_136'][i] = hand
            
        self.game_state['remaining_tiles'] -= (13 * 4 + 14) # 嶺上牌などを除く (ドラ表示、裏ドラ、嶺上牌)

    def _draw_tile(self, player):
        """ツモ処理"""
        if not self.game_state['total_tiles']:
            return False # 牌山切れ
            
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
        if tile in self.game_state['hands_136'][player]:
            self.game_state['hands_136'][player].remove(tile)
        else:
             if self.game_state['last_drawn_tile'][player] is not None:
                 tile = self.game_state['last_drawn_tile'][player]
                 self.game_state['last_drawn_tile'][player] = None
        
        self.game_state['rivers'][player].append(tile)
        # 他家からのロン/鳴き判定のため、直近の捨て牌を記録
        self.game_state['last_discarded_tile'] = tile
        self.game_state['last_discarder'] = player


    def _execute_tsumo_agari(self, player):
        """ツモ和了処理と報酬計算"""
        win_tile = self.game_state['last_drawn_tile'][player]
        result = self._calculate_hand_value(player, win_tile, is_tsumo=True)
        
        if result and result.error is None:
            reward = self._get_tsumo_reward(result.cost, player)
            self._add_event('AGARI', {'winner': player, 'from': player, 'cost': result.cost})
            return reward, True
        else:
            return -1000.0, False

    def _execute_ron_agari(self, player, winning_tile):
        """ロン和了処理と報酬計算"""
        result = self._calculate_hand_value(player, winning_tile, is_tsumo=False)
        
        if result and result.error is None:
            discarder = self.game_state.get('last_discarder', (player - 1) % 4)
            reward = self._get_ron_reward(result.cost, player, discarder)
            self._add_event('AGARI', {'winner': player, 'from': discarder, 'cost': result.cost})
            return reward, True
        else:
            return -1000.0, False
            
    def _calculate_hand_value(self, player, win_tile, is_tsumo):
        """HandCalculatorを呼び出し、点数結果オブジェクトを返す"""
        hand_136 = self.game_state['hands_136'][player]
        
        config = HandConfig(
            is_tsumo=is_tsumo,
            player_wind=Meld.EAST + player, 
            round_wind=Meld.EAST + self.game_state['round'] % 4,
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
        """ツモ和了時の報酬計算 (点棒移動)"""
        oya = self.game_state['oya_player_id']
        # 報酬は、獲得点数 - 支払点数。ここでは獲得点数を単純化して返す
        
        if winner_id == oya: # 親のツモ
             payment_sum = cost['main'] * 3 + cost['additional'] * 3 + self.game_state['honba'] * 300 * 3
        else: # 子のツモ
             payment_sum = cost['main'] + cost['additional'] * 2 + self.game_state['honba'] * 300
                 
        return payment_sum

    def _get_ron_reward(self, cost, winner_id, loser_id):
        """ロン和了時の報酬計算 (点棒移動)"""
        main_score = cost['main'] * 2 + cost['additional']
        payment = main_score + self.game_state['honba'] * 300
        
        return payment

    def _handle_termination(self, forced_termination):
        """局の終了処理 (流局判定、点数移動)"""
        
        if self.game_state['remaining_tiles'] <= 0 and not forced_termination:
            self._add_event('RYUKYOKU', {'reason': 'Haishan Kire'})
            return 0.0, True
            
        return 0.0, True

    def _add_event(self, event_id, data):
        """ゲームイベントを記録 (Parserの_add_eventに相当)"""
        event = {'event_id': event_id, 'turn': self.game_state['turn_num']}
        event.update(data)
        self.game_state['events'].append(event)
    
    # ----------------------------------------------------------------------
    # 行動・観測関連
    # ----------------------------------------------------------------------

    def _decode_action(self, action_id):
        """行動IDから、牌ID (またはタイプ) とアクションタイプをデコード"""
        if 0 <= action_id < TILE_34_COUNT:
            return action_id, 'DISCARD' 
        
        action_map = {
            34: 'PASS', 35: 'TSUMO_AGARI', 36: 'RON_AGARI', 37: 'RIICHI',
            38: 'PUNG', 39: 'CHII', 40: 'DAIMINKAN', 41: 'ANKAN', 42: 'KAKAN'
        }
        
        return None, action_map.get(action_id, 'PASS')

    def _get_action_mask(self, player):
        """合法的な行動を1、非合法的な行動を0とするマスクを生成"""
        
        mask = np.zeros(TOTAL_ACTION_DIM, dtype=np.int8)
        
        mask[34] = 1 # 1. パスは常に可能

        # 2. 打牌 (34種)
        hand_136 = self.game_state['hands_136'][player]
        hand_34 = TilesConverter.to_34_array(hand_136)
        unique_tiles_34 = np.where(hand_34 > 0)[0]
        for tile_34 in unique_tiles_34:
             mask[tile_34] = 1 
             
        # 3. ツモ和了判定 (35)
        last_drawn = self.game_state['last_drawn_tile'][player]
        if last_drawn is not None and self._can_agari_at_this_moment(player, last_drawn, is_tsumo=True):
            mask[35] = 1 # ACTION_TSUMO_AGARI

        # 4. リーチ判定 (37)
        if not self.game_state['is_riichi'][player]:
            shanten = self.shanten_calculator.calculate_shanten(hand_34)
            if shanten <= 0:
                 mask[37] = 1 # RIICHI (宣言)

        # TODO: その他の特殊行動の判定 (鳴き、暗槓、加槓)

        return mask

    def _get_observation(self, player):
        """現在の状態をTransformerの入力形式に変換 (仕様書に準拠)"""
        
        # 1. Context (イベントシーケンスのベクトル化)
        context_vector = np.zeros((CONTEXT_MAX_LEN, EVENT_VECTOR_DIM), dtype=np.float32)
        
        # 2. Action Mask
        action_mask = self._get_action_mask(player)
        
        return {"context": context_vector, "action_mask": action_mask}

    def _can_agari_at_this_moment(self, player, win_tile, is_tsumo):
        """mahjongライブラリを用いて和了可能かを確認"""
        result = self._calculate_hand_value(player, win_tile, is_tsumo)
        return result and result.error is None
        
    def render(self, mode="human"):
        """対局の状況をコンソールに表示 (デバッグ用)"""
        if mode == "human":
            print("--- Mahjong Game State ---")
            print(f"Round: {self.game_state['round'] + 1} ({['東', '南', '西', '北'][self.game_state['round'] % 4]}{self.game_state['round'] % 4 + 1}局), Honba: {self.game_state['honba']}")
            print(f"Current Player: P{self.current_player} ({['東', '南', '西', '北'][self.current_player]}) (RL Agent ID: {self.rl_agent_id})")
            
            hand_136 = self.game_state['hands_136'][self.current_player]
            hand_34_array = TilesConverter.to_34_array(hand_136)
            hand_string = TilesConverter.to_one_line_string(hand_34_array)
            shanten = self.shanten_calculator.calculate_shanten(hand_34_array)
            print(f"Hand: {hand_string} (Shanten: {shanten})")
            print(f"Remaining Tiles: {self.game_state['remaining_tiles']}")

    def close(self):
        """環境のクリーンアップ"""
        pass

# ----------------------------------------------------------------------
# 動作確認 (Colabでの実行時、この部分は不要)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    env = MahjongEnv(agent_id=0)
    obs, info = env.reset()
    
    print("Environment Reset. Initial Observation:")
    env.render()
    
    # RLエージェント(P0)がツモ切りを試行 (行動ID 0は萬子の1)
    action_to_take = 0
    print(f"\nRL Agent P{env.rl_agent_id} takes Action ID: {action_to_take} (Mask: {obs['action_mask'][action_to_take]})")
    
    obs, reward, terminated, truncated, info = env.step(action_to_take)
    
    print(f"Reward: {reward}, Terminated: {terminated}")
    env.render()
    
    if not terminated:
        print(f"\nNext Player is RL Agent P{env.current_player} again (after 3 Bot turns).")
        env.render()
        
    env.close()
