import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
from datasets import load_dataset, Dataset
from rewards import SYSTEM_PROMPT

DATA_DIR = os.path.join(f"/scratch/{os.environ['USER']}", "rl-chess", "data")
SFT_GENERAL = 15_000
SFT_MATING = 10_000
SFT_ENDGAME = 5_000
GRPO_SIZE = 5_000
MIN_RATING = 1200
MAX_RATING = 2200

MATING_THEMES = {"mateIn1", "mateIn2", "mateIn3", "backRankMate", "smotheredMate"}
ENDGAME_THEMES = {"endgame", "pawnEndgame", "rookEndgame", "knightEndgame", "bishopEndgame", "promotion"}


def process_puzzle(row):
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    if len(moves) < 2:
        return None
    try:
        board.push(chess.Move.from_uci(moves[0]))
    except (ValueError, chess.InvalidMoveError):
        return None
    puzzle_fen = board.fen()
    best_move = moves[1]
    try:
        if chess.Move.from_uci(best_move) not in board.legal_moves:
            return None
    except (ValueError, chess.InvalidMoveError):
        return None
    legal = " ".join(m.uci() for m in board.legal_moves)
    return {
        "fen": puzzle_fen,
        "best_move": best_move,
        "legal_moves": legal,
        "rating": row["Rating"],
    }


def build_sft_messages(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Position (FEN): {example['fen']}\n"
                    f"Legal moves: {example['legal_moves']}"
                ),
            },
            {"role": "assistant", "content": f"<move>{example['best_move']}</move>"},
        ]
    }


def build_grpo_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Position (FEN): {example['fen']}\n"
                    f"Legal moves: {example['legal_moves']}"
                ),
            },
        ],
        "fen": example["fen"],
    }


def collect_puzzles(dataset, theme_filter, limit):
    """Collect puzzles matching a theme filter."""
    results = []
    for row in dataset:
        raw = row.get("Themes", "")
        themes = set(raw) if isinstance(raw, list) else set(raw.split())
        if theme_filter(themes):
            result = process_puzzle(row)
            if result is not None:
                results.append(result)
            if len(results) >= limit:
                break
    return results


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Loading Lichess puzzles dataset...")
    ds = load_dataset("Lichess/chess-puzzles", split="train")
    print(f"Total puzzles: {len(ds)}")

    filtered = ds.filter(
        lambda x: MIN_RATING <= x["Rating"] <= MAX_RATING,
        num_proc=4,
    )
    print(f"Filtered to {MIN_RATING}-{MAX_RATING} rating: {len(filtered)}")

    filtered = filtered.shuffle(seed=42)

    # Collect diverse puzzle types
    print(f"Collecting {SFT_MATING} mating puzzles...")
    mating = collect_puzzles(filtered, lambda t: bool(t & MATING_THEMES), SFT_MATING)
    print(f"  Got {len(mating)} mating puzzles")

    print(f"Collecting {SFT_ENDGAME} endgame puzzles...")
    endgame = collect_puzzles(filtered, lambda t: bool(t & ENDGAME_THEMES), SFT_ENDGAME)
    print(f"  Got {len(endgame)} endgame puzzles")

    # Use FENs from themed puzzles to avoid duplicates in general set
    themed_fens = {p["fen"] for p in mating + endgame}
    print(f"Collecting {SFT_GENERAL + GRPO_SIZE} general puzzles...")
    general = collect_puzzles(
        filtered,
        lambda t: True,  # any theme
        SFT_GENERAL + GRPO_SIZE + len(themed_fens),  # over-collect to skip dupes
    )
    general = [p for p in general if p["fen"] not in themed_fens]
    print(f"  Got {len(general)} general puzzles (deduplicated)")

    # Build SFT dataset: general + mating + endgame
    sft_data = general[:SFT_GENERAL] + mating + endgame
    grpo_data = general[SFT_GENERAL : SFT_GENERAL + GRPO_SIZE]

    print(f"\nSFT data: {len(sft_data)} total ({SFT_GENERAL} general + {len(mating)} mating + {len(endgame)} endgame)")
    print(f"GRPO data: {len(grpo_data)} examples")

    sft_dataset = Dataset.from_list([build_sft_messages(ex) for ex in sft_data])
    sft_path = os.path.join(DATA_DIR, "sft_dataset")
    sft_dataset.save_to_disk(sft_path)
    print(f"SFT dataset saved to {sft_path} ({len(sft_dataset)} examples)")

    grpo_dataset = Dataset.from_list([build_grpo_example(ex) for ex in grpo_data])
    grpo_path = os.path.join(DATA_DIR, "grpo_dataset")
    grpo_dataset.save_to_disk(grpo_path)
    print(f"GRPO dataset saved to {grpo_path} ({len(grpo_dataset)} examples)")


if __name__ == "__main__":
    main()
