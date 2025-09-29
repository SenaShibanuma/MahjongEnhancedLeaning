# -*- coding: utf-8 -*-
from mahjong.tile import TilesConverter
import numpy as np

# ----------------------------------------------------------------------
# 定数定義 (environment.pyと同期)
# ----------------------------------------------------------------------
TILE_34_COUNT = 34 
CONTEXT_MAX_LEN = 50 
EVENT_VECTOR_DIM = 256 # environment.pyのMahjongEnvで定義されたサイズ

# ----------------------------------------------------------------------
# ユーティリティ関数
# ----------------------------------------------------------------------

def convert_tile_136_to_34(tile_136_id):
    """136牌IDを34種牌IDに変換 (0-33)"""
    # 0-3萬 -> 0, 4-7萬 -> 1, ...
    return tile_136_id // 4

def _encode_hand_tiles(hand_136):
    """手牌を34種の枚数ベクトル (0-4) に変換"""
    return TilesConverter.to_34_array(hand_136)

def _generate_event_vector(event, current_game_state, rl_agent_id):
    """
    単一のゲームイベントとゲーム状態から、固定長のイベントベクトルを生成する。
    
    【実装方針】: あなたの仕様書 (AgentInterfaceSpecification) の EventVector に基づく。
    この関数は、イベントの種類にかかわらず、256次元のベクトルを返す必要があります。
    """
    
    vector = np.zeros(EVENT_VECTOR_DIM, dtype=np.float32)
    
    # ------------------------------------------------------------
    # A. 共通パラメータのエンコーディング (仕様書 CommonParameters)
    # ------------------------------------------------------------
    offset = 0
    
    # 1. Player_ID (相対位置: 0=自分, 1=下家, 2=対面, 3=上家)
    player_id = event.get('player', -1)
    relative_id = (player_id - rl_agent_id + 4) % 4 if player_id != -1 else -1
    if relative_id != -1:
        vector[offset + relative_id] = 1.0 # 4次元One-Hot
    offset += 4
    
    # 2. Turn_Counter (巡目)
    vector[offset] = current_game_state.get('turn_num', 0) / 30.0 # 正規化 (最大30巡程度を想定)
    offset += 1
    
    # 3. Scores (4人全員の現在点)
    scores = np.array(current_game_state['scores']) / 30000.0 # 30000点で正規化
    # スコアはプレイヤー視点に回転させる
    rotated_scores = np.roll(scores, -rl_agent_id)
    vector[offset:offset + 4] = rotated_scores
    offset += 4
    
    # 4. Riichi_Sticks (供託リーチ棒の数)
    vector[offset] = current_game_state.get('riichi_sticks', 0) / 5.0 # 最大5本として正規化
    offset += 1
    
    # 5. Remaining_Tiles_Count (牌山の残り枚数)
    vector[offset] = current_game_state.get('remaining_tiles', 0) / 70.0 # 70枚で正規化
    offset += 1
    
    # 6. Visible_Tiles_Vector (全見え牌の34枚数ベクトル)
    visible_counts = np.zeros(TILE_34_COUNT)
    
    # 河の牌
    for player_river in current_game_state['rivers']:
        for tile_136 in player_river:
            visible_counts[convert_tile_136_to_34(tile_136)] += 1
            
    # ★修正: 副露牌 (melds) の加算
    for player_id, player_melds in enumerate(current_game_state['melds']):
        # RLエージェント自身の手牌は、手牌ベクトル (7) で考慮されるため、ここでは含めない。
        # ただし、副露面子は既に手牌から除かれている前提のため、すべて含める。
        # mahjong.Meldオブジェクトにはtilesプロパティがある
        for meld in player_melds:
            for tile_136 in meld.tiles:
                visible_counts[convert_tile_136_to_34(tile_136)] += 1
            
    # ドラ表示牌 
    for indicator_136 in current_game_state['dora_indicators']:
        visible_counts[convert_tile_136_to_34(indicator_136)] += 1
    
    vector[offset:offset + TILE_34_COUNT] = visible_counts
    offset += TILE_34_COUNT
    
    # ------------------------------------------------------------
    # B. プレイヤー固有情報 (自分視点 - このイベントの観測時)
    # ------------------------------------------------------------
    
    # 7. 自分の手牌 (34枚数ベクトル)
    my_hand_136 = current_game_state['hands_136'][rl_agent_id]
    my_hand_34 = _encode_hand_tiles(my_hand_136)
    vector[offset:offset + TILE_34_COUNT] = my_hand_34
    offset += TILE_34_COUNT

    # 8. 自分のシャンテン数 (このイベントの前の状態を反映しているはず)
    # Note: MahjongEnvでシャンテン数が状態に保存されていないため、ここではダミー
    shanten = 9
    # try:
    #     shanten = self.shanten_calculator.calculate_shanten(_encode_hand_tiles(my_hand_136))
    # except:
    #     pass
    vector[offset] = shanten / 9.0
    offset += 1
    
    # ------------------------------------------------------------
    # C. イベント固有情報のエンコーディング (仕様書 EventTypes)
    # ------------------------------------------------------------
    
    EVENT_ID_MAP = {'INIT': 0, 'DRAW': 1, 'DISCARD': 2, 'MELD': 3, 'RIICHI': 4, 'AGARI': 5, 'RYUKYOKU': 6, 'GAME_START': 7}
    EVENT_TYPE_ONE_HOT_DIM = 8 
    
    # 9. イベントタイプ One-Hot
    event_id_idx = EVENT_ID_MAP.get(event['event_id'], -1)
    if event_id_idx >= 0 and event_id_idx < EVENT_TYPE_ONE_HOT_DIM:
        vector[offset + event_id_idx] = 1.0
    offset += EVENT_TYPE_ONE_HOT_DIM
    
    # 10. 牌情報 (打牌/ツモ牌)
    tile_136 = event.get('tile') # DISCARD/DRAW イベント用
    if tile_136 is not None:
        tile_34 = convert_tile_136_to_34(tile_136)
        vector[offset + tile_34] = 1.0 # 34次元One-Hot
    offset += TILE_34_COUNT
    
    # 11. 特殊フラグ (is_tedashi, is_red_dora, etc.)
    # is_tedashi (DISCARDイベント用)
    vector[offset] = event.get('is_tedashi', 0.0) 
    offset += 1
    
    # 喰いタン/赤ドラフラグ (INIT/GAME_STARTイベント用)
    if event['event_id'] == 'GAME_START':
        vector[offset] = event['rules'].get('has_kuitan', 0.0)
        vector[offset+1] = event['rules'].get('has_aka_dora', 0.0)
    offset += 2
    
    # ... 残り次元 (256 - offset) はゼロパディング
    
    return vector

def get_context_vector(game_state, rl_agent_id):
    """
    ゲーム状態からTransformerの入力となるContextベクトルシーケンスを生成する
    """
    events = game_state['events']
    context_list = []
    
    for event in events:
        # 各イベントをベクトル化
        event_vector = _generate_event_vector(event, game_state, rl_agent_id)
        context_list.append(event_vector)
        
    # NumPy配列に変換
    context_np = np.array(context_list, dtype=np.float32)
    
    # 最大長にパディング/切り捨て
    if context_np.shape[0] > CONTEXT_MAX_LEN:
        context_vector = context_np[-CONTEXT_MAX_LEN:, :] # 古いイベントを切り捨て
    else:
        # 短い場合は、ゼロベクトルでパディングする (先頭にパディング)
        padding_size = CONTEXT_MAX_LEN - context_np.shape[0]
        padding = np.zeros((padding_size, EVENT_VECTOR_DIM), dtype=np.float32)
        context_vector = np.vstack((padding, context_np))

    # 形状: (CONTEXT_MAX_LEN, EVENT_VECTOR_DIM)
    return context_vector
