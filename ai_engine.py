# ai_engine.py
import chess
import random

def get_ai_move(board):
    """
    简单的 AI：随机选择一个合法走法。
    """
    legal_moves = list(board.legal_moves)
    if legal_moves:
        return random.choice(legal_moves)
    return None
