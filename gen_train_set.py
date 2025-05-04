import io
import chess.pgn
import pandas as pd
from encode import create_board_planes
import numpy as np
from static_evaluate import evaluate

GAMMA = 0.995     # 终局折扣率
ALPHA = 0.5       # 基础权重（0 全部靠阶段，1 全部靠差值）
MIN_DELTA = 5     # 差值阈值，小于则视为平庸
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
        temp_Y = []
        for index, move in enumerate(game.mainline_moves()):
            encoded = create_board_planes(board)
            x_train.append(encoded)

            # 1. 静态评估差值
            before = evaluate(board)
            board.push(move)
            after = evaluate(board)
            delta = after - before
            # 2. 阈值剪枝
            if abs(delta) < MIN_DELTA:
                delta = 0.0
            # 3. 线性阶段因子 φ = t/T
            phi = index / float(total_moves)
            # 4. 阶段加权
            local = delta * (ALPHA + (1 - ALPHA) * phi)
            # 5. 折扣到当前步：γ^(T-t)
            discount = GAMMA ** (total_moves - index)
            raw = local * discount

            # y_t = BETA * res_value + (1 - BETA) * raw
            max_abs_value = max(max_abs_value, abs(raw))
            temp_Y.append(raw)

        y_train.extend([x / max_abs_value for x in temp_Y]) # 归一化处理
        print(f"parsing game {game_cnt}, got {len(y_train)} examples")

        if samples is not None and len(y_train) > samples:
            break
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

if __name__ == "__main__":
    x_train, y_train = get_data_set(10000)
    np.savez("data/dataset_10k_without_res.npz", x_train, y_train)