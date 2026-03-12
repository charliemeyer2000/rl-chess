# RL Chess

Demo notebook file of teaching a 7B model to beat ChatGPT 4o in chess for the UVA AI Safety Club

## Setup

### What you need to bring

- Environment Variables (see .env.example):
    - OPENAI_API_KEY
    - HF_TOKEN
    - WANDB_API_KEY

- If you're using UVA HPC cluster to train, we using the `rv` cli (documentation [here](https://rivanna.dev)). If you are willing to spend some money, we highly suggest using [modal](https://modal.com/) for serverless GPUs and easy management.

### Coding Environment

- Ensure you have some basics set up:
    - Python (we suggest managing python versions + environments with [`uv`](https://docs.astral.sh/uv/))
    - Git (comes pre-installed on most computers, but just check with `git --version`), and a GitHub account.
    - Some sort of IDE
    - Some sort of AI assistance (we suggest [claude code](https://code.claude.com/docs/en/overview)), but you can get access to various AI tools for free as a student:
        - [Gemini for Students](https://gemini.google/students/)
        - [Cursor for Students](https://cursor.com/students)
        - [Windsurf for Students](https://windsurf.com/editor/students)
    - If you don't have a huggingface account, please make one and an token.
    - If you don't have a weights and biases account, please make one (and sign up with your .edu email for their student plan) and make an api key.
    - If you don't have an openai platform account, make one and create an api key.

## Overview

What we're attempting to do here is teach some small language model, in this case, `Qwen-2.5-7B-Instruct` to beat ChatGPT 4o in a game of chess. Most language models aren't particularly good at playing chess — they'll hallucinate, make illegal moves, or just don't know enough chess strategy to win. We fix this with a 3-phase training pipeline.

### Strategy

1. **Data Prep** — Download Lichess puzzles (rated 1200-2200), extract positions + best moves, and create 15K SFT + 5K GRPO training examples.
2. **SFT (Supervised Fine-Tuning)** — Teach the model our chess prompt format (FEN + legal moves → `<move>` tags) and basic strategy using LoRA. Takes ~1.5h on 1x A100.
3. **GRPO (Reinforcement Learning)** — Have the model play against itself, scored by a progressive reward function (format → legality → Stockfish quality). Uses vLLM in colocate mode on 1x A100, ~20-30 min for 500 steps.
4. **Evaluation** — Test on held-out puzzles (legal move rate, Stockfish score) and play full games vs GPT-4o.
5. **Play Against It** — Serve the model with vLLM and play in the browser via the Next.js web app in `web/`.

### Results

- **100% legal move rate** on 100 held-out puzzles
- **~1100 avg centipawn** Stockfish score
- **90% win rate vs GPT-4o** (9W/0L/1D — most wins from GPT-4o making illegal moves)

### Project Structure

```
scripts/
  data_prep.py        # Download + process Lichess puzzles into SFT/GRPO datasets
  sft_train.py        # SFT with LoRA on Qwen 2.5 7B
  grpo_train.py       # GRPO training with vLLM colocate mode
  run_grpo.sh         # Wrapper: installs vLLM, finds SFT checkpoint, launches GRPO
  rewards.py          # Reward functions: format, legality, quality (Stockfish)
  evaluate.py         # Puzzle eval + full games vs GPT-4o
  setup_stockfish.sh  # Build Stockfish from source on the cluster
  serve_model.sh      # Start vLLM OpenAI-compatible server for inference
web/                  # Next.js app to play chess against the model in the browser
```

### Demo Notebooks

- `rivanna-run.ipynb` — example of this with UVA HPC (using `rv` cli)
- `modal-run.ipynb` — example of this with modal
- `cluster.ipynb` — example of this on a regular node (assuming some GPU-accelerated node, detects cuda/mps)

### Playing Against Your Model

After training, you can play against the model in the browser:

```bash
# Terminal 1: Start vLLM on Rivanna
rv run -g 1 -t a100 --time 3h --name chess-serve bash scripts/serve_model.sh

# Terminal 2: Forward port (use job ID from `rv ps`)
rv forward 8000 <JOB_ID>

# Terminal 3: Start the web app
cd web && pnpm install && pnpm dev
# Open http://localhost:3000
```
