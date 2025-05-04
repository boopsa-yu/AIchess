import io
import chess.pgn
import pandas as pd
from state import State

# get chess data with elo >= 2000
chess_data_test = pd.read_csv('data/chess_data_2000.csv')
for pgn_text in chess_data_test[:1]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    result = game.headers["Result"]
    value = {'1/2-1/2':0, '1-0':1, '0-1':-1}[result]
    # print(value)

    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        print(State(board).create_board_planes())
