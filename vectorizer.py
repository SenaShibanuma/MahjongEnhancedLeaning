# -*- coding: utf-8 -*-
import numpy as np

# ----------------------------------------------------------------------
# 定数定義 (プロジェクト全体で共有)
# ----------------------------------------------------------------------
TILE_34_COUNT = 34
EVENT_VECTOR_DIM = 128  # 観測ベクトルの次元数
CONTEXT_MAX_LEN = 50    # 観測シーケンスの最大長

# --- 行動空間 (Action Space) の定義 ---
# AIが取りうる全ての行動を文字列のリストとして定義する
# このリストのインデックスが、AIの出力 (action_id) となる
ACTION_SPACE = (
    # 1. 打牌 (34種)
    [f"DISCARD_{i}" for i in range(TILE_34_COUNT)] +
    # 2. 特殊アクション
    [
        "ACTION_RIICHI",
        "ACTION_TSUMO_AGARI",
        "ACTION_PASS", # 合法手がない場合の安全策
        # TODO: 将来的にはチー、ポン、カンなどの鳴きアクションも追加
    ]
)

# ----------------------------------------------------------------------
# 観測ベクトル生成関数
# ----------------------------------------------------------------------
def get_observation_vector(game_state_for_vectorizer):
    """
    ゲームロジックから渡された状態辞書を、
    Transformerモデルへの入力となる固定長のベクトルシーケンスに変換する。
    """
    events = game_state_for_vectorizer['events']
    context_list = []

    for event in events:
        event_vec = _generate_single_event_vector(event, game_state_for_vectorizer)
        context_list.append(event_vec)

    # --- 状態ベクトルを追加 ---
    # 現在の手牌、ドラ、点数などの「静的な」状態を表すベクトルを追加
    state_vec = _generate_current_state_vector(game_state_for_vectorizer)
    context_list.append(state_vec)

    # NumPy配列に変換し、パディング/切り捨て
    context_np = np.array(context_list, dtype=np.float32)
    
    if context_np.shape[0] > CONTEXT_MAX_LEN:
        context_vector = context_np[-CONTEXT_MAX_LEN:, :]
    else:
        padding_size = CONTEXT_MAX_LEN - context_np.shape[0]
        padding = np.zeros((padding_size, EVENT_VECTOR_DIM), dtype=np.float32)
        context_vector = np.vstack((padding, context_np))
    
    return context_vector

def _generate_single_event_vector(event, full_state):
    """個別のイベント（打牌、ツモなど）をベクトル化する"""
    vec = np.zeros(EVENT_VECTOR_DIM, dtype=np.float32)
    # (イベントのベクトル化ロジック ... )
    # 例:
    # if event['event_id'] == 'DISCARD':
    #     vec[0] = 1.0 
    #     player_offset = (event['player'] - full_state['player_id'] + 4) % 4
    #     vec[1 + player_offset] = 1.0
    #     tile_34 = event['tile'] // 4
    #     vec[5 + tile_34] = 1.0
    return vec

def _generate_current_state_vector(full_state):
    """現在の静的な盤面状態をベクトル化する"""
    vec = np.zeros(EVENT_VECTOR_DIM, dtype=np.float32)
    offset = 0

    # 自分の手牌 (34次元)
    my_hand_34 = np.bincount(
        [t // 4 for t in full_state['my_hand_136']], 
        minlength=TILE_34_COUNT
    )
    vec[offset : offset + TILE_34_COUNT] = my_hand_34
    offset += TILE_34_COUNT

    # ドラ表示牌 (34次元 One-Hot)
    for dora_indicator in full_state['dora_indicators']:
        dora_34 = dora_indicator // 4
        vec[offset + dora_34] = 1.0
    offset += TILE_34_COUNT

    # 点数 (4次元 正規化)
    scores = np.array(full_state['scores']) / 50000.0 # 50000点で正規化
    vec[offset : offset + 4] = scores
    offset += 4
    
    # 巡目 (1次元 正規化)
    vec[offset] = full_state['turn'] / 20.0 # 20巡で正規化
    offset += 1
    
    # 向聴数 (1次元 正規化)
    vec[offset] = full_state['shanten'] / 8.0 # 8向聴で正規化
    offset += 1

    return vec

