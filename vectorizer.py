# -*- coding: utf-8 -*-
"""
AIに与える「観測データ」を生成する共通モジュール。
教師あり学習と強化学習の両方で、このファイルの関数を使用します。
"""
import numpy as np
from mahjong.tile import TilesConverter

# --- システム全体で共有する定数 ---
# 牌の種類 (萬子1-9, 筒子1-9, 索子1-9, 字牌7)
TILE_34_COUNT = 34
# 行動の定数
ACTION_DISCARD_OFFSET = 0
ACTION_RIICHI = 34
ACTION_AGARI_TSUMO = 35
ACTION_PASS = 36 # 将来の鳴き判断などで使用

# 行動の総数: 打牌(34), リーチ(1), ツモ和了(1), パス(1)
TOTAL_ACTION_DIM = 37
# Transformerが読み込むイベント履歴の最大長
CONTEXT_MAX_LEN = 50
# 各イベントを表すベクトルの次元数
EVENT_VECTOR_DIM = 128

def convert_tile_136_to_34(tile_136):
    """136 ID (0-135) を 34種 ID (0-33) に変換"""
    if tile_136 is None or not (0 <= tile_136 < 136):
        return -1 # 無効な牌
    return tile_136 // 4

def get_context_vector(game_state, rl_agent_id):
    """
    ゲームのイベント履歴から、AIへの入力となる状況(Context)ベクトルを生成する。
    """
    game_events = game_state.get('events', [])
    event_vectors = []
    # 常に最新のイベントから CONTEXT_MAX_LEN 個だけを対象にする
    start_index = max(0, len(game_events) - CONTEXT_MAX_LEN)
    relevant_events = game_events[start_index:]

    for event in relevant_events:
        event_vector = _generate_event_vector(event, game_state, rl_agent_id)
        event_vectors.append(event_vector)

    # NumPy配列に変換
    context_np = np.array(event_vectors, dtype=np.float32)

    # 短い場合はゼロでパディング (前方パディング)
    if context_np.shape[0] < CONTEXT_MAX_LEN:
        padding_size = CONTEXT_MAX_LEN - context_np.shape[0]
        padding = np.zeros((padding_size, EVENT_VECTOR_DIM), dtype=np.float32)
        padded_context = np.vstack([padding, context_np])
    else:
        padded_context = context_np

    return padded_context

def _generate_event_vector(event, game_state, rl_agent_id):
    """単一のイベントを固定長のベクトルに変換する"""
    vector = np.zeros(EVENT_VECTOR_DIM, dtype=np.float32)
    offset = 0

    # --- イベント自体の情報 ---
    # イベントタイプ (8次元 One-Hot)
    event_types = {'INIT': 0, 'DRAW': 1, 'DISCARD': 2, 'RIICHI': 3, 'AGARI': 4}
    event_idx = event_types.get(event.get('event_id'), -1)
    if event_idx != -1:
        vector[offset + event_idx] = 1.0
    offset += 8

    # イベントの主体プレイヤー (自分, 下家, 対面, 上家の4次元 One-Hot)
    player_id = event.get('player', -1)
    if player_id != -1:
        relative_id = (player_id - rl_agent_id + 4) % 4
        vector[offset + relative_id] = 1.0
    offset += 4

    # 関連する牌 (34次元 One-Hot)
    tile_136 = event.get('tile', None)
    tile_34 = convert_tile_136_to_34(tile_136)
    if tile_34 != -1:
        vector[offset + tile_34] = 1.0
    offset += TILE_34_COUNT # 34

    # --- イベント発生時の全体状況 ---
    # 巡目 (正規化)
    vector[offset] = game_state.get('turn_num', 0) / 20.0 # 20巡を最大と仮定
    offset += 1

    # 各プレイヤーの点数 (自分視点に回転させ、正規化)
    scores = np.array(game_state.get('scores', [25000]*4))
    rotated_scores = np.roll(scores, -rl_agent_id)
    vector[offset : offset+4] = rotated_scores / 35000.0 # 35000点を基準
    offset += 4

    # リーチ棒の数 (正規化)
    vector[offset] = game_state.get('riichi_sticks', 0) / 4.0
    offset += 1

    # ドラ表示牌 (34次元 One-Hot)
    dora_indicators = game_state.get('dora_indicators', [])
    if dora_indicators:
        dora_34 = convert_tile_136_to_34(dora_indicators[0])
        if dora_34 != -1:
            vector[offset + dora_34] = 1.0
    offset += TILE_34_COUNT # 34

    # 各プレイヤーのリーチ状態 (自分視点に回転)
    is_riichi = np.array(game_state.get('is_riichi', [False]*4), dtype=np.float32)
    rotated_riichi = np.roll(is_riichi, -rl_agent_id)
    vector[offset : offset+4] = rotated_riichi
    offset += 4

    # 残りベクトル長: 128 - (8+4+34+1+4+1+34+4) = 38
    # 残りの次元はゼロパディングされる

    return vector

