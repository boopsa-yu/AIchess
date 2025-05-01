# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import chess
from ai_engine import get_ai_move

# 创建 FastAPI 应用实例
app = FastAPI()
# URL 路径映射，用于前端资源
app.mount("/static", StaticFiles(directory="static"), name="static")
# 指定模板目录
templates = Jinja2Templates(directory="templates")

# 初始化棋局
board = chess.Board()

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


# 定义对 GET / 的请求处理器，返回内容是 HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 走子处理接口
# 玩家选择棋子移动，之后AI根据玩家的选择移动
@app.post("/move")
async def move(request: Request):
    global board
    if board.is_game_over():
        return JSONResponse(
            content={ 'status': 'finished', 'result': board.result() },
            status_code=200
        )

    data = await request.json()
    source = data.get('from')
    target = data.get('to')
    promotion = data.get('promotion', 'q') # 默认升变为 后

    try:
        move = chess.Move.from_uci(source + target)
        if move not in board.legal_moves:
            return JSONResponse(content={'error': '非法走法'}, status_code=400)
    except ValueError:
        return JSONResponse(content={'error': '非法走法'}, status_code=400)

    board.push(move)
    # 检查玩家走子后是否结束
    result_info = get_game_result(board)
    if result_info["game_over"]:
        return {
            "ai_move": None,
            **result_info
        }

    # AI 走法
    ai_move = get_ai_move(board)
    if ai_move:
        board.push(ai_move)
        # 判定是否和棋
        result_info = get_game_result(board)
        return {
            "ai_move": ai_move.uci(),
            **result_info
        }
    else:
        result_info = get_game_result(board)
        return {
            "ai_move": None,
            **result_info
        }

@app.post("/reset")
async def reset():
    global board
    board.reset()
    return {'status': '重置成功'}
