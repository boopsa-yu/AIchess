import chess
import numpy as np

def create_board_planes(board: chess.Board):
    """
    序列化，用于模型输入
    square: 棋盘上的格子，返回值为一个 0~63 的整数
    """
    assert board.is_valid()
    piece2num = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                 "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}

    board_state = np.zeros(64, np.uint8)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            board_state[i] = piece2num[piece.symbol()]

    if board.ep_square is not None:
        assert board_state[board.ep_square] == 0 # 空格，否则无法跨过
        board_state[board.ep_square] = 8 # 将潜在的吃路过兵的格设定为 8
    # 王车易位权利
    if board.has_queenside_castling_rights(chess.WHITE):
        assert board_state[0] == 4 and board_state[4] == 6
        board_state[0] = 7
    if board.has_kingside_castling_rights(chess.WHITE):
        assert board_state[7] == 4 and board_state[4] == 6
        board_state[7] = 7
    if board.has_queenside_castling_rights(chess.BLACK):
        assert board_state[56] == 8 + 4 and board_state[60] == 8 + 6
        board_state[56] = 8 + 7
    if board.has_kingside_castling_rights(chess.BLACK):
        assert board_state[63] == 8 + 4 and board_state[60] == 8 + 6
        board_state[63] = 8 + 7
    
    board_state = board_state.reshape(8, 8)

    # binary state
    state = np.zeros((5,8,8), np.uint8)

    # 0-3 columns to binary
    state[0] = (board_state >> 3) & 1
    state[1] = (board_state >> 2) & 1
    state[2] = (board_state >> 1) & 1
    state[3] = (board_state >> 0) & 1

    # 第 4 列用于判断当前的回合
    state[4] = 1.0 if board.turn == chess.WHITE else 0.0

    return state