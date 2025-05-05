import io
import chess.pgn
import numpy as np
import pandas as pd
from encode import create_board_planes
import static_evaluate
from matplotlib import pyplot as plt

ALPHA = 0.5       # 基础权重，0 表示全部靠阶段
BETA = 0.5        # 混合比例

# get chess data with elo >= 2000
def get_data_set(samples=None):
    x_train = []
    y_train = []
    game_cnt = 0
    value = {'1/2-1/2':0, '1-0':1, '0-1':-1}

    chess_data_test = pd.read_csv('data/chess_data_2000.csv')
    for row, pgn_text in chess_data_test.itertuples():
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        # result = game.headers["Result"]
        # if result not in value:
        #     # add log
        #     continue
        # res_value = value[result]
        game_cnt += 1

        board = game.board()
        half_moves = list(game.mainline_moves())
        # 每 2 个 half-move 算作 1 个 full-move
        total_moves = (len(half_moves) + 1) // 2
        max_abs_value = 0.0
        y_temp = []
        for index, move in enumerate(game.mainline_moves()):
            encoded = create_board_planes(board)
            x_train.append(encoded)

            # 1. 静态评估
            static_value = static_evaluate.evaluate(board)
            # 2. 材料相位 φ
            phi = static_evaluate.phase(board)
            # 3. 阶段加权 Tapered Eval
            raw = static_value * (ALPHA + (1 - ALPHA) * phi)

            # y_t = BETA * res_value + (1 - BETA) * raw
            max_abs_value = max(max_abs_value, abs(raw))
            y_temp.append(raw)
            board.push(move)

        # 4. 标签数据归一化处理
        y_norm = [x / max_abs_value for x in y_temp]
        # TODO 考虑最终结果的影响

        y_train.extend(y_norm)
        print(f"parsing game {game_cnt}, got {len(y_train)} examples")

        if samples is not None and len(y_train) > samples:
            break
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

if __name__ == "__main__":
    x_train, y_train = get_data_set(5000000)
    np.savez("data/dataset_5M_without_res.npz", x_train, y_train)