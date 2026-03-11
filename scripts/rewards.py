import re
import math
import os
import atexit
import shutil

import chess
import chess.engine

SYSTEM_PROMPT = (
    "You are a chess engine. Given the current board position in FEN notation "
    "and the list of legal moves, select the best move in UCI notation.\n\n"
    "Respond with your chosen move inside <move> tags. "
    "For example: <move>e2e4</move>\n\n"
    "You may optionally reason about the position before providing your move."
)

_engine = None


def find_stockfish():
    for path in [
        os.environ.get("STOCKFISH_PATH", ""),
        os.path.expanduser("~/.local/bin/stockfish"),
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]:
        if path and os.path.isfile(path):
            return path
    path = shutil.which("stockfish")
    if path:
        return path
    raise FileNotFoundError(
        "Stockfish not found. Run: rv run --mig bash scripts/setup_stockfish.sh"
    )


def get_engine():
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    return _engine


def _cleanup():
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        _engine = None


atexit.register(_cleanup)


def _text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def extract_move(text):
    match = re.search(r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>", text)
    if match:
        return match.group(1)
    return None


def format_reward(completions, **kwargs):
    return [1.0 if extract_move(_text(c)) is not None else 0.0 for c in completions]


def legality_reward(completions, fen, **kwargs):
    rewards = []
    for completion, f in zip(completions, fen):
        move_str = extract_move(_text(completion))
        if not move_str:
            rewards.append(0.0)
            continue
        try:
            board = chess.Board(f)
            move = chess.Move.from_uci(move_str)
            rewards.append(1.0 if move in board.legal_moves else 0.0)
        except (ValueError, chess.InvalidMoveError):
            rewards.append(0.0)
    return rewards


def quality_reward(completions, fen, **kwargs):
    engine = get_engine()
    rewards = []
    for completion, f in zip(completions, fen):
        move_str = extract_move(_text(completion))
        if not move_str:
            rewards.append(0.0)
            continue
        try:
            board = chess.Board(f)
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                rewards.append(0.0)
                continue
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=12))
            cp = info["score"].relative.score(mate_score=10000)
            rewards.append(1.0 / (1.0 + math.exp(0.004 * cp)))
        except (ValueError, chess.InvalidMoveError):
            rewards.append(0.0)
    return rewards


def make_prompt(fen):
    board = chess.Board(fen)
    legal = " ".join(m.uci() for m in board.legal_moves)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Position (FEN): {fen}\nLegal moves: {legal}",
        },
    ]
