import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import chess.engine
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewards import find_stockfish, extract_move, SYSTEM_PROMPT

OUTPUT_DIR = os.environ.get(
    "RV_OUTPUT_DIR",
    os.path.join(f"/scratch/{os.environ['USER']}", ".rv", "outputs", "chess-eval"),
)


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_model_move(model, tokenizer, fen):
    board = chess.Board(fen)
    legal = " ".join(m.uci() for m in board.legal_moves)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Position (FEN): {fen}\nLegal moves: {legal}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_move(response), response


def get_gpt4o_move(client, fen):
    board = chess.Board(fen)
    legal = " ".join(m.uci() for m in board.legal_moves)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Position (FEN): {fen}\nLegal moves: {legal}"},
        ],
        max_tokens=256,
        temperature=0.3,
    )
    text = response.choices[0].message.content
    return extract_move(text), text


def evaluate_move(engine, board, move_uci):
    if move_uci is None:
        return {"legal": False, "score": None}
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return {"legal": False, "score": None}
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        score = info["score"].relative.score(mate_score=10000)
        board.pop()
        return {"legal": True, "score": score}
    except (ValueError, chess.InvalidMoveError):
        return {"legal": False, "score": None}


def play_game(model, tokenizer, client, engine, max_moves=80):
    board = chess.Board()
    our_color = chess.WHITE
    moves = []

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        fen = board.fen()
        if board.turn == our_color:
            move_uci, _ = get_model_move(model, tokenizer, fen)
        else:
            move_uci, _ = get_gpt4o_move(client, fen)

        if move_uci is None:
            result = "illegal_move"
            moves.append({"fen": fen, "move": None, "player": "us" if board.turn == our_color else "gpt4o"})
            break

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                result = "illegal_move"
                moves.append({"fen": fen, "move": move_uci, "player": "us" if board.turn == our_color else "gpt4o", "illegal": True})
                break
            board.push(move)
            moves.append({"fen": fen, "move": move_uci, "player": "us" if board.turn == our_color else "gpt4o"})
        except (ValueError, chess.InvalidMoveError):
            result = "illegal_move"
            break
    else:
        result = "draw_by_length"

    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner == our_color:
            result = "win"
        elif outcome.winner is None:
            result = "draw"
        else:
            result = "loss"

    return {
        "result": result,
        "num_moves": len(moves),
        "moves": moves,
        "final_fen": board.fen(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model (default: GRPO merged checkpoint)",
    )
    parser.add_argument("--num_puzzles", type=int, default=100)
    parser.add_argument("--num_games", type=int, default=10)
    args = parser.parse_args()

    if args.model_path is None:
        checkpoints_base = os.path.join(
            f"/scratch/{os.environ['USER']}", ".rv", "checkpoints"
        )
        for name in ["chess-grpo", "chess-sft"]:
            path = os.path.join(checkpoints_base, name, "merged")
            if os.path.exists(path):
                args.model_path = path
                break
        if args.model_path is None:
            args.model_path = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Evaluating model: {args.model_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    model, tokenizer = load_model(args.model_path)

    from datasets import load_from_disk
    data_dir = os.path.join(f"/scratch/{os.environ['USER']}", "rl-chess", "data")
    grpo_dataset = load_from_disk(os.path.join(data_dir, "grpo_dataset"))

    print(f"\n=== Puzzle Evaluation ({args.num_puzzles} positions) ===")
    legal_count = 0
    quality_scores = []

    for i, example in enumerate(grpo_dataset.select(range(min(args.num_puzzles, len(grpo_dataset))))):
        fen = example["fen"]
        board = chess.Board(fen)
        move_uci, _ = get_model_move(model, tokenizer, fen)
        result = evaluate_move(engine, board, move_uci)

        if result["legal"]:
            legal_count += 1
        if result["score"] is not None:
            quality_scores.append(result["score"])

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{args.num_puzzles}] Legal: {legal_count}/{i+1} ({100*legal_count/(i+1):.1f}%)")

    puzzle_results = {
        "legal_rate": legal_count / args.num_puzzles,
        "avg_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        "num_evaluated": args.num_puzzles,
    }
    print(f"\nLegal move rate: {puzzle_results['legal_rate']:.1%}")
    print(f"Avg Stockfish score: {puzzle_results['avg_score']:.1f} cp")

    client = OpenAI()
    print(f"\n=== Games vs GPT-4o ({args.num_games} games) ===")
    game_results = []

    for i in range(args.num_games):
        print(f"  Game {i+1}/{args.num_games}...", end=" ", flush=True)
        result = play_game(model, tokenizer, client, engine)
        game_results.append(result)
        print(f"{result['result']} ({result['num_moves']} moves)")

    wins = sum(1 for g in game_results if g["result"] == "win")
    losses = sum(1 for g in game_results if g["result"] == "loss")
    draws = sum(1 for g in game_results if g["result"] in ("draw", "draw_by_length"))
    gpt4o_illegals = sum(1 for g in game_results if g["result"] == "illegal_move" and g.get("moves", [{}])[-1].get("player") == "gpt4o")

    summary = {
        "model_path": args.model_path,
        "puzzle_results": puzzle_results,
        "game_results": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "gpt4o_illegal_moves": gpt4o_illegals,
            "total": args.num_games,
            "win_rate": wins / args.num_games,
        },
    }

    print("\n=== Summary ===")
    print(f"Puzzles: {puzzle_results['legal_rate']:.1%} legal, {puzzle_results['avg_score']:.1f} avg cp")
    print(f"vs GPT-4o: {wins}W / {losses}L / {draws}D (win rate: {wins/args.num_games:.1%})")

    results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    engine.quit()


if __name__ == "__main__":
    main()
