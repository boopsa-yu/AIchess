import torch
import chess
import numpy as np
from cnn import CNN
from encode import create_board_planes

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

def get_best_moves(board: chess.Board, eval):
    res = []
    turn = board.turn
    for move in board.legal_moves:
        board.push(move)
        res.append((move, eval(board)))
        board.pop()

    return sorted(res, key=lambda x: x[1], reverse=turn)[:3]

# if __name__ == "__main__":
#     eval = BoardEvaluator("models/model0.1.pth")
#     board = chess.Board()
#     while not board.is_game_over():
#         moves = get_best_moves(board, eval)
#         print(moves[0])
#         move = moves[0][0]
#         board.push(move)
#     print(board.result())

