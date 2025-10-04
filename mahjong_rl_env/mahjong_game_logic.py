# -*- coding: utf-8 -*-
import random
import json
import numpy as np # NumPyをインポート
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.meld import Meld

# 外部のベクトル化定義をインポート
from vectorizer import ACTION_SPACE, TILE_34_COUNT

class MahjongGameLogic:
    """
    TF-Agents環境から独立した、純粋な4人麻雀のゲームロジック。
    状態管理、行動の実行、合法手判定、報酬計算などを担当する。
    """
    def __init__(self):
        self.shanten_calculator = Shanten()
        self.hand_calculator = HandCalculator()
        self.game_state = {}
        self.rule_config = HandConfig(options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))

    def reset(self):
        """新しい局を開始する"""
        self._initialize_new_round()
        self._deal_tiles()
        
        # 親が最初のツモ
        self.game_state['current_player_id'] = self.game_state['oya_player_id']
        drawn_tile = self._draw_tile(self.game_state['current_player_id'])

        # 初回のイベントを記録
        self._add_event('START_GAME', {})
        self._add_event('DRAW', {'player': self.game_state['current_player_id'], 'tile': drawn_tile})

        return self.game_state

    def step(self, player_id, action_id):
        """
        指定されたプレイヤーの行動を実行し、ゲームを1ステップ進める
        戻り値: (次のゲーム状態, 報酬, 局が終了したか, {追加情報})
        """
        action_name = ACTION_SPACE[action_id]
        reward = 0
        is_terminated = False
        info = {'yaku': None} # 和了役や流局理由を格納

        # --- 1. 行動の実行 ---
        if action_name.startswith('DISCARD_'):
            tile_34 = int(action_name.split('_')[1])
            self._execute_discard(player_id, tile_34)
            
            # 他プレイヤーの応答チェック（ロン、ポンなど）
            # (ここに他プレイヤーの応答判定ロジックを実装)
            # ...

            # 応答がなければ、次のプレイヤーがツモ
            next_player = (player_id + 1) % 4
            drawn_tile = self._draw_tile(next_player)
            self.game_state['current_player_id'] = next_player
            self._add_event('DRAW', {'player': next_player, 'tile': drawn_tile})


        elif action_name == 'ACTION_TSUMO_AGARI':
            reward, yaku_result = self._execute_agari(player_id, is_tsumo=True)
            info['yaku'] = yaku_result
            is_terminated = True

        elif action_name == 'ACTION_RIICHI':
            # (リーチ宣言の処理)
            pass

        # --- 2. 局の終了判定 ---
        if self.game_state['remaining_tiles'] <= 0:
            is_terminated = True
            info['yaku'] = "RYUKYOKU_HAITEI" # 流局（海底）

        # --- 3. 牌譜の記録 ---
        self.game_state['kifu'].append({
            'player': player_id,
            'action': action_name,
            'state_after': self._get_kifu_state() # 行動後の状態を記録
        })

        return self.game_state, reward, is_terminated, info

    def get_action_mask(self, player_id):
        """プレイヤーが現在取れる合法な行動のマスクを返す"""
        # --- 修正箇所: データ型を int8 から int32 に変更 ---
        mask = np.zeros(len(ACTION_SPACE), dtype=np.int32)
        hand_136 = self.game_state['hands_136'][player_id]
        
        # NumPy配列に変換してから処理
        hand_34 = TilesConverter.to_34_array(hand_136)
        
        # 1. 打牌可能な牌をマスク=1にする
        # last_drawn_tileも手牌に含めて考える
        if self.game_state['last_drawn_tile'][player_id] is not None:
            drawn_tile_34 = self.game_state['last_drawn_tile'][player_id] // 4
            hand_34[drawn_tile_34] += 1

        for tile_34, count in enumerate(hand_34):
            if count > 0:
                action_idx = ACTION_SPACE.index(f"DISCARD_{tile_34}")
                mask[action_idx] = 1

        # 2. ツモ和了可能かチェック
        win_tile = self.game_state['last_drawn_tile'][player_id]
        if win_tile is not None and self._can_agari(player_id, win_tile, is_tsumo=True):
            action_idx = ACTION_SPACE.index("ACTION_TSUMO_AGARI")
            mask[action_idx] = 1
        
        # 3. リーチ可能かチェック
        # (向聴数が0で、特定の条件を満たした場合にマスク=1にする)
        # ...

        # 合法手が一つもない場合（ありえないが安全策）、ダミーの行動（パスなど）を許可
        if np.sum(mask) == 0:
            mask[ACTION_SPACE.index("ACTION_PASS")] = 1

        return mask

    def get_state_for_vectorizer(self, player_id):
        """vectorizer.pyが要求する形式で現在の状態を返す"""
        return {
            'events': self.game_state['events'],
            'my_hand_136': self.game_state['hands_136'][player_id],
            'dora_indicators': self.game_state['dora_indicators'],
            'scores': self.game_state['scores'],
            'turn': self.game_state['turn_num'],
            'player_id': player_id,
            'shanten': self._get_shanten(player_id)
        }
        
    def get_kifu(self):
        """現在の局の牌譜データを返す"""
        return {
            'initial_state': self.game_state['initial_kifu_state'],
            'actions': self.game_state['kifu']
        }

    # ======================================================================
    # 内部ヘルパー関数
    # ======================================================================

    def _initialize_new_round(self):
        """局の内部状態をリセット"""
        all_tiles = list(range(136))
        random.shuffle(all_tiles)

        self.game_state = {
            'events': [], 'round': 0, 'honba': 0, 'riichi_sticks': 0,
            'dora_indicators': [all_tiles.pop(0)],
            'scores': [25000] * 4,
            'hands_136': [sorted(all_tiles[i*13:(i+1)*13]) for i in range(4)],
            'melds': [[] for _ in range(4)],
            'rivers': [[] for _ in range(4)], 'is_riichi': [False] * 4,
            'oya_player_id': random.randint(0, 3),
            'turn_num': 0,
            'last_drawn_tile': [None] * 4,
            'remaining_tiles': 136 - (13 * 4) - 14, # 王牌
            'wall': all_tiles[13*4:],
            'current_player_id': 0,
            'kifu': [],
            'initial_kifu_state': None, # 初期状態を保存
        }
        # 初期状態の牌譜は _get_kifu_state を呼び出す前に設定
        self.game_state['initial_kifu_state'] = self._get_kifu_state(is_initial=True)

    def _deal_tiles(self):
        # _initialize_new_round に統合
        pass

    def _draw_tile(self, player_id):
        """牌山から牌を1枚引き、手牌に加え、引いた牌を返す"""
        if not self.game_state['wall']:
            return None
        drawn_tile = self.game_state['wall'].pop(0)
        # self.game_state['hands_136'][player_id].append(drawn_tile) # 手牌にはまだ加えない
        self.game_state['last_drawn_tile'][player_id] = drawn_tile
        self.game_state['remaining_tiles'] -= 1
        self.game_state['turn_num'] += 1
        return drawn_tile

    def _execute_discard(self, player_id, tile_34):
        """打牌処理"""
        hand = self.game_state['hands_136'][player_id]
        drawn_tile = self.game_state['last_drawn_tile'][player_id]

        # ツモ切りか手出しかを判定
        discard_tile_136 = None
        if drawn_tile is not None and (drawn_tile // 4) == tile_34:
            discard_tile_136 = drawn_tile
            self.game_state['last_drawn_tile'][player_id] = None # ツモ牌を消費
        else:
            # 手の中から該当する牌を探して捨てる
            for tile in hand:
                if tile // 4 == tile_34:
                    discard_tile_136 = tile
                    hand.remove(tile)
                    break
        
        if discard_tile_136 is not None:
            # ツモ牌を手牌に加える
            if drawn_tile is not None and drawn_tile != discard_tile_136:
                 hand.append(drawn_tile)
                 self.game_state['last_drawn_tile'][player_id] = None
            
            hand.sort()
            self.game_state['rivers'][player_id].append(discard_tile_136)
            self._add_event('DISCARD', {'player': player_id, 'tile': discard_tile_136})
        else:
            # 不正な打牌（エラーハンドリング）
            print(f"警告: P{player_id} が不正な打牌 {tile_34} を試みました。")

    def _can_agari(self, player_id, win_tile, is_tsumo):
        """指定された牌で和了可能か判定"""
        hand_136 = self.game_state['hands_136'][player_id][:]
        if is_tsumo:
            # 手牌にツモ牌を加えた状態で判定
            hand_136.append(win_tile)

        try:
            result = self.hand_calculator.estimate_hand_value(
                tiles=hand_136,
                win_tile=win_tile,
                melds=self.game_state['melds'][player_id],
                dora_indicators=self.game_state['dora_indicators'],
                config=self._get_config(player_id, is_tsumo)
            )
            return result.error is None
        except Exception:
            return False

    def _execute_agari(self, player_id, is_tsumo, win_tile=None):
        """和了処理と点数計算"""
        if is_tsumo:
            win_tile = self.game_state['last_drawn_tile'][player_id]
        
        hand_136 = self.game_state['hands_136'][player_id][:]
        hand_136.append(win_tile)

        result = self.hand_calculator.estimate_hand_value(
            tiles=hand_136,
            win_tile=win_tile,
            melds=self.game_state['melds'][player_id],
            dora_indicators=self.game_state['dora_indicators'],
            config=self._get_config(player_id, is_tsumo)
        )
        
        # 点数移動と報酬計算
        cost = result.cost['main'] + result.cost['additional']
        yaku_names = ", ".join([y.name for y in result.yaku])
        
        self._add_event('AGARI', {'player': player_id, 'score': cost, 'yaku': yaku_names})
        
        # (点数移動のロジックをここに実装)
        # self.game_state['scores'][player_id] += cost
        # ...

        return cost, yaku_names
    
    def _get_shanten(self, player_id):
        """指定プレイヤーの現在の向聴数を計算"""
        hand_136 = self.game_state['hands_136'][player_id]
        melds = self.game_state['melds'][player_id]
        hand_34 = TilesConverter.to_34_array(hand_136)
        
        # calculate_shanten は、七対子・国士無双も考慮して最小向聴数を返す
        shanten_result = self.shanten_calculator.calculate_shanten(hand_34, melds=melds)
        return shanten_result

    def _get_config(self, player_id, is_tsumo):
        """点数計算用の設定オブジェクトを生成"""
        player_wind = Meld.EAST + ((player_id - self.game_state['oya_player_id'] + 4) % 4)
        round_wind = Meld.EAST + (self.game_state['round'] // 4)
        
        return HandConfig(
            is_tsumo=is_tsumo,
            player_wind=player_wind,
            round_wind=round_wind,
            is_riichi=self.game_state['is_riichi'][player_id],
            options=self.rule_config.options
        )

    def _add_event(self, event_id, data):
        """ゲームイベントを履歴に追加"""
        event = {'event_id': event_id, 'turn': self.game_state['turn_num']}
        event.update(data)
        self.game_state['events'].append(event)
        
    def _get_kifu_state(self, is_initial=False):
        """牌譜記録用に、現在の盤面状態をJSON化可能な形式で返す"""
        # is_initialに応じてhandsの形式を変える
        if is_initial:
            # 配牌は136形式でそのまま記録
            hands_data = [self.game_state['hands_136'][i] for i in range(4)]
        else:
            # 途中経過は34種牌の枚数配列（リスト）で記録
            hands_data = [TilesConverter.to_34_array(h).tolist() for h in self.game_state['hands_136']]
            
        state = {
            'scores': self.game_state['scores'],
            'dora_indicators': self.game_state['dora_indicators'],
            'turn': self.game_state['turn_num'],
            'current_player_id': self.game_state['current_player_id'],
            'hands': hands_data,
            'rivers': [[t // 4 for t in r] for r in self.game_state['rivers']],
        }
        return state

