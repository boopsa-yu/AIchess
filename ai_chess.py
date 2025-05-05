import torch
import chess
import numpy as np
from cnn import CNN
from encode import create_board_planes

MAXVAL = 10000

class BoardEvaluator:
    def __init__(self, model_path: str):
        self.model = CNN()
        model_dict = torch.load(model_path)
        self.model.load_state_dict(model_dict)


    def __call__(self, board: chess.Board):
        planes = create_board_planes(board)
        input = planes[np.newaxis, :]  # 适配神经网络输入 1 * 5*8*8
        output = self.model(torch.tensor(input).float())
        return output.item()
    
    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
        if depth == 0 or board.is_game_over():
            if board.is_game_over():
                if board.is_checkmate():
                    return (-MAXVAL if board.turn == chess.WHITE else MAXVAL)
                else:
                    return 0  # draw
            return self(board)

        best_value = -float('inf') if maximizing_player else float('inf')

        for move in board.legal_moves:
            board.push(move)
            value = self.minimax(board, depth - 1, alpha, beta, not maximizing_player)
            board.pop()

            if maximizing_player:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)

            if beta <= alpha:
                break  # α-β 剪枝

        return best_value
    
    def get_best_moves(self, board: chess.Board, depth: int = 2):
        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            score = self.minimax(board, depth - 1, -float('inf'), float('inf'), board.turn == chess.WHITE)
            board.pop()

            if board.turn == chess.WHITE and score > best_score:
                best_score = score
                best_move = move
            elif board.turn == chess.BLACK and score < best_score:
                best_score = score
                best_move = move

        return best_move, best_score
