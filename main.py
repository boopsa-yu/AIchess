# main.py
from time import sleep
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import chess
from ai_engine import get_ai_move
import logging

# 创建 FastAPI 应用实例
app = FastAPI()
# URL 路径映射，用于前端资源
app.mount("/static", StaticFiles(directory="static"), name="static")
# 指定模板目录
templates = Jinja2Templates(directory="templates")

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)  # 将日志输出至屏幕


# 初始化棋局
board = chess.Board()
players = {
    chess.WHITE: "human",
    chess.BLACK: "ai"
}

# 定义对 GET / 的请求处理器，返回内容是 HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 走子处理接口
# 玩家选择棋子移动，之后AI根据玩家的选择移动
@app.post("/move/{player}")
async def move(request: Request, player: str):
    global board
    data = await request.json() if player == "human" else None

    moves = {
        "human": human_move,
        "ai": ai_move
    }
    
    if player not in moves:
        return JSONResponse(
            content={'error': '非法的移动类型'},
            status_code=400
        )

    return one_move_step(moves[player], data)

@app.post("/reset")
async def reset():
    global board
    board.reset()
    return {'status': '重置成功',
            'current_player': players.get(board.turn)
    }


def get_game_result(board: chess.Board) -> dict:
    """
    检查棋局结束状态并返回结果字典：
    - game_over: bool
    - result: "1-0" / "0-1" / "1/2-1/2" / " Playing "
    - reason: "checkmate" / "stalemate" / "insufficient_material" / ...
    """
    if not board.is_game_over():
        return {"game_over": False, "result": "", "reason": ""}
    # 以下按优先级判断具体结束方式
    if board.is_checkmate():
        # 轮到哪方走但被将死，则对方获胜
        winner = "Black" if board.turn == chess.WHITE else "White"
        score = "1-0" if winner == "White" else "0-1"
        return {"game_over": True, "result": score, "reason": "checkmate"}
    if board.is_stalemate():
        # 逼和
        return {"game_over": True, "result": "1/2-1/2", "reason": "stalemate"}
    if board.is_insufficient_material():
        # 死局
        return {"game_over": True, "result": "1/2-1/2", "reason": "insufficient_material"}
    if board.is_seventyfive_moves() or board.is_fivefold_repetition():
        # 75步和棋规则和五次重复局面和棋规则
        return {"game_over": True, "result": "1/2-1/2", "reason": "draw_by_rules"}
    # 其他结束情况
    return {"game_over": True, "result": "1/2-1/2", "reason": "other"}

def human_move(data) -> chess.Move:
    global board
    if 'from' not in data or 'to' not in data:
        raise ValueError("缺少移动数据")
    
    uci_str = f"{data['from']}{data['to']}{data.get('promotion', '')}"
    move = chess.Move.from_uci(uci_str)
    if move not in board.legal_moves:
        raise ValueError("非法走法")
    return move

def ai_move(data=None) -> chess.Move:
    """AI移动生成逻辑"""
    global board
    move = get_ai_move(board)
    sleep(1)
    if not move:
        raise RuntimeError("AI无法生成合法移动")
    return move

def one_move_step(move_generator, data) -> dict:
    """通用移动处理框架"""
    global board

    # 验证当前回合类型
    current_player = players.get(board.turn)
    expected_player_type = "human" if move_generator == human_move else "ai"
    if current_player != expected_player_type:
        return JSONResponse(
            content={'error': f'当前不是 {expected_player_type} 回合'},
            status_code=400
        )

    # 生成移动
    try:
        move = move_generator(data)  # data参数需要根据移动类型传递
    except Exception as e:
        return JSONResponse(
            content={'error': str(e)},
            status_code=400 if isinstance(e, ValueError) else 500
        )

    # 执行移动并返回状态
    board.push(move)
    result_info = get_game_result(board)  # 状态检查
    return {
        "fen": board.fen(),
        "last_move": move.uci(),
        **result_info,
        "next_player": players.get(board.turn)
    }
