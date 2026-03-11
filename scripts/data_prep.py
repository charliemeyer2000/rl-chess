import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
from datasets import load_dataset, Dataset
from rewards import SYSTEM_PROMPT

DATA_DIR = os.path.join(f"/scratch/{os.environ['USER']}", "rl-chess", "data")
SFT_SIZE = 15_000
GRPO_SIZE = 5_000
MIN_RATING = 1200
MAX_RATING = 2200


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

    print("Processing puzzles...")
    processed = []
    for row in filtered:
        result = process_puzzle(row)
        if result is not None:
            processed.append(result)
        if len(processed) >= SFT_SIZE + GRPO_SIZE:
            break

    print(f"Processed {len(processed)} valid puzzles")

    sft_data = processed[:SFT_SIZE]
    grpo_data = processed[SFT_SIZE : SFT_SIZE + GRPO_SIZE]

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
