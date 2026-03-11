# Training a 7B LLM to beat GPT-4o at chess with GRPO

**A Qwen 2.5 7B model fine-tuned with supervised learning and GRPO reinforcement learning can realistically beat GPT-4o at chess—because GPT-4o is surprisingly bad at it.** GPT-4o achieves roughly ~1540 Elo at best and makes illegal moves ~13% of the time, while multiple 2025 research projects have already demonstrated 7B models reaching competitive chess play through SFT+GRPO pipelines. The optimal stack for a single-notebook educational demo is TRL's `GRPOTrainer` with vLLM on Modal, using python-chess and Stockfish for reward signals, at an estimated cost of **$15–25** for the full training run. The critical insight from recent research: SFT warmstart is non-negotiable—RL alone plateaus because pretrained LLMs lack sufficient internal chess representations.

---

## GPT-4o is a surprisingly weak chess opponent

Beating GPT-4o at chess is a lower bar than most people expect. Benchmarks across multiple evaluation frameworks paint a consistent picture: **GPT-4o achieves roughly 1540 Elo with optimal prompting** (using the "regurgitation + examples" trick documented by Dynomight), but in standard play it struggles to complete full games and frequently makes illegal moves. On the LLM Chess leaderboard (50+ models tested), GPT-4o couldn't consistently beat even a random player in 2024, with most games ending in 200-move draws.

The irony is that the older **gpt-3.5-turbo-instruct model (~1750 Elo)** dramatically outperforms all chat-tuned models at chess, because it was trained on raw PGN game data via text completion rather than RLHF-tuned for conversation. Chat/instruct tuning appears to actively degrade chess ability. OpenAI's reasoning models (o1, o3, GPT-5) represent a qualitative leap, but GPT-4o specifically remains weak. A 7B model that can reliably generate legal moves and make decent positional choices would beat it handily.

For a puzzle-solving benchmark, GPT-4o solved 501 out of 1000 Lichess puzzles correctly (with a 12.7% illegal move rate). This means a 7B model trained to play chess at even a modest club level (~1200–1500 Elo) would likely surpass GPT-4o on game play, and certainly on puzzle accuracy with near-zero illegal moves.

---

## Chess over checkers, despite the complexity gap

Chess is the right game for this demo, despite checkers being simpler. The deciding factors are ecosystem maturity and prior work, not game complexity.

Chess offers **massive training data** (Lichess's database contains 1.7B+ games), a world-class evaluation engine (Stockfish), a mature Python library (python-chess), and extensive LLM-specific research to build on. At least six major projects in 2025 trained LLMs on chess with GRPO or similar RL methods. By contrast, virtually no LLM RL work exists for checkers—no equivalent Stockfish for shaped rewards, no large-scale game datasets, and no reference implementations.

The action space difference (chess ~30 legal moves per position vs. checkers ~7) matters less than it appears, because GRPO naturally handles larger action spaces through group sampling. Move representation is straightforward in both cases: **UCI notation** (`e2e4`) is the standard for LLM chess, offering fixed-length 4-character strings that are unambiguous, directly compatible with python-chess, and easily validated with a simple regex (`^[a-h][1-8][a-h][1-8][qrbn]?$`). Board state uses FEN strings in prompts.

| Factor | Chess | Checkers |
|--------|-------|----------|
| Available training data | 1.7B+ games (Lichess) | Very limited |
| Evaluation engine | Stockfish (world-class) | No equivalent |
| Existing LLM RL research | 6+ major projects (2025) | Virtually none |
| Python library | python-chess (mature) | Limited options |
| Action space per position | ~30 legal moves | ~7 legal moves |
| Game tree complexity | ~10^123 | ~10^31 |

---

## Six existing projects provide a proven roadmap

The landscape of LLM chess RL research in 2025 is remarkably rich. These projects collectively establish the recipe for training a 7B model:

**Xiangqi-R1** is the most directly relevant precedent. It trains Qwen-2.5-7B-Instruct to play Chinese chess via a 3-stage pipeline: SFT for legal move prediction on 5M board-move pairs, strategic annotation incorporation, then GRPO with multi-dimensional rewards. Running on 4× A6000 GPUs, it achieves **95.2% legal move accuracy** and significantly outperforms much larger general LLMs (including 671B models) on spatial reasoning tasks.

**Chess-R1** (Krafton AI) tested GRPO directly on Qwen2.5-3B and 7B for chess, with both sparse binary and dense Stockfish-based rewards. Their critical finding: **all models plateau at ~25–30% puzzle accuracy**, well below expert level (~66.5% for 1800 Elo). Dense rewards outperform sparse rewards, but RL alone cannot overcome the pretrained model's lack of deep chess understanding. This sets realistic expectations for the demo.

**"Reasoning Through Chess"** (NeurIPS 2025 Workshop) achieved the most impressive results: a Qwen2.5-7B-Instruct model that **surpasses a 120B-parameter model** on chess benchmarks after SFT on 120M tokens followed by RL. Their key insight: SFT data recipe determines RL ceiling. "Best Move" SFT enables effective RL training, while "Best Line" SFT gives faithful reasoning traces.

**GRPO&Master** (Stanford CS224R) decomposed chess into six subtasks (legal moves, piece counting, capture identification, etc.) and trained Qwen-2.5-3B with multi-task GRPO, achieving **7.2× improvement** over baseline with just 3,000 training examples.

**rezabonyadi/chess-language** provides the closest template for the demo: a single Jupyter notebook implementing GRPO chess training with a progressive multi-step reward function (format compliance → move validity → legality → Stockfish evaluation) and XML-structured chain-of-thought reasoning.

The **Reasoning Gym** (100+ procedural data generators with verifiable rewards) and **verifiers** library (GRPO trainer with multi-turn environment support) provide additional infrastructure for building game environments compatible with LLM RL training.

---

## TRL is the clear framework choice for a single-notebook demo

After comparing TRL, OpenRLHF, and veRL across six dimensions, **TRL (Hugging Face) wins decisively** for an educational, single-GPU context. The comparison is not close.

TRL's `GRPOTrainer` requires roughly **10 lines of Python** for a basic training loop. It natively supports vLLM in "colocate" mode—where the inference engine shares GPU memory with the trainer—enabling single-A100 GRPO training of 7B models. Custom reward functions are plain Python callables passed directly to the trainer. The documentation is outstanding: a dedicated HF LLM Course chapter, multiple Cookbook notebooks, and official Qwen model examples.

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    num_generations=8,  # Group size
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    report_to="wandb",
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    reward_funcs=[format_reward, legality_reward, quality_reward],
    train_dataset=chess_dataset,
    args=config,
)
trainer.train()
```

OpenRLHF and veRL are designed for multi-GPU production workloads. OpenRLHF requires Ray clusters and Docker, driven by ~40-line bash scripts with CLI flags—not notebook-friendly. veRL uses Hydra config with ~40+ parameters and FSDP for distributed training. Both can technically run on a single A100 with LoRA, but neither is documented or tested for that scenario.

| Criterion | TRL | OpenRLHF | veRL |
|-----------|-----|----------|------|
| Code for basic loop | ~10 lines Python | ~40 lines bash | ~30 lines config |
| Single A100 80GB (7B) | ✅ Documented | ⚠️ Untested | ⚠️ Feasible with LoRA |
| Installation | `pip install trl[vllm]` | Docker recommended | Docker recommended |
| Beginner-friendliness | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Custom reward functions | Python callable | Remote RM URL file | Custom reward file |
| Notebook-friendly | Yes | No | Partial |

**Unsloth** deserves mention as a TRL wrapper that reduces VRAM by ~80% via custom Triton kernels. It enables 7B GRPO training on GPUs as small as 16GB. For an A100 80GB, vanilla TRL is sufficient and simpler, but Unsloth is a viable option if targeting more accessible hardware (e.g., Colab T4).

---

## The reward function is the heart of the demo

The reward function design is both the most educational and most impactful component. A **progressive multi-component reward** is the proven approach, building from format compliance up to move quality:

```python
def chess_reward(completions, board_fens, **kwargs):
    rewards = []
    for completion, fen in zip(completions, board_fens):
        board = chess.Board(fen)
        reward = 0.0
        
        # Step 1: Format compliance (+0.1)
        move_str = extract_move_from_xml(completion)
        if move_str:
            reward += 0.1
        
            # Step 2: Valid UCI format (+0.1)
            try:
                move = chess.Move.from_uci(move_str)
                reward += 0.1
                
                # Step 3: Legal move (+0.3)
                if move in board.legal_moves:
                    reward += 0.3
                    
                    # Step 4: Stockfish quality (+0.0 to 0.5)
                    board.push(move)
                    info = engine.analyse(board, chess.engine.Limit(depth=15))
                    cp = info["score"].relative.score(mate_score=10000)
                    reward += 0.5 * sigmoid(0.004 * cp)
                    board.pop()
            except ValueError:
                pass
        rewards.append(reward)
    return rewards
```

This graduated approach is critical because a 7B model initially cannot make legal chess moves at all. GRPO generates G=8–16 rollouts per position; with graduated rewards, even partially correct outputs (right format, parseable but illegal move) receive some signal, enabling learning from the start. Binary win/loss rewards provide too sparse a signal for initial training.

**Including the list of legal moves in the prompt** is the single most impactful intervention for reducing illegal moves, based on findings across multiple projects. The prompt template should include the FEN position, legal moves in UCI format, and any previous move history.

For handling illegal moves during training, **GRPO's natural rejection mechanism** is elegant: illegal-move completions receive low rewards, get negative advantages in the group, and are down-weighted in the policy update. No explicit masking or retry logic is needed—the algorithm handles it naturally.

---

## The end-to-end pipeline in practice

### Stage 1: SFT warmstart (essential, ~1–2 hours)

SFT warmstart is non-negotiable for chess. Every successful project uses it. The pretrained Qwen 2.5 7B has minimal chess knowledge—RL cannot teach spatial reasoning from scratch, only amplify existing capabilities. Fine-tune on **5K–20K examples** of FEN → UCI move pairs from high-Elo games (the `angeluriot/chess_games` HuggingFace dataset with 14M games is ideal, filtered to 2000+ Elo). Use **LoRA** (rank 16, alpha 32) to reduce memory. Target: **>90% legal move rate** before starting RL.

### Stage 2: GRPO training (~4–8 hours on single A100)

Use TRL's `GRPOTrainer` with vLLM colocate mode. Key hyperparameters based on existing projects:

- **Group size (G)**: 8 (stable, memory-efficient; G=8 has "virtually no impact on GPU memory" per TRL docs)
- **KL coefficient (β)**: 0.0 (TRL default; skips reference model, saving ~14GB VRAM)
- **Learning rate**: 1e-6 to 5e-6
- **Clip ratio (ε)**: 0.2
- **Max completion length**: 256–512 tokens (allows chain-of-thought reasoning)
- **Training data**: 2K–5K Lichess puzzle positions with clear best moves

With vLLM colocate on a single A100 80GB: 5K prompts × 8 generations × ~100 tokens/generation ≈ 4M tokens. At ~500 tokens/sec vLLM throughput, generation alone takes ~2.2 hours. With training overhead, expect **4–8 hours total** for a meaningful GRPO run.

### Stage 3: Evaluation

Metrics to track via wandb (natively integrated with TRL via `report_to="wandb"`):
- Legal move rate (should approach 99%+ after SFT+RL)
- Mean Stockfish evaluation of chosen moves
- Reward component breakdown (format, legality, quality)
- KL divergence from reference policy

### Infrastructure and cost

**Modal** is the recommended compute platform. It has an **official GRPO training example** (`modal.com/docs/examples/grpo_trl`) using TRL + vLLM, with wandb integration via Modal Secrets and checkpoint storage via Modal Volumes. A100 80GB costs **~$2.50/hour** on Modal with per-second billing and $30/month free credits. Total estimated cost: **$15–25** for a complete SFT + GRPO training run.

The full infrastructure stack:

- **Compute**: Modal (1× A100 80GB)
- **Training**: TRL `GRPOTrainer` + vLLM colocate
- **Game engine**: python-chess + Stockfish
- **Model**: `Qwen/Qwen2.5-7B-Instruct` with LoRA
- **Logging**: Weights & Biases
- **Data**: `angeluriot/chess_games` or Lichess puzzle dataset on HuggingFace
- **Package management**: UV (`uv add trl[vllm] python-chess wandb stockfish`)

---

## Structuring the notebook for maximum educational impact

The minimum viable demo is a **next-move prediction task** (FEN → best UCI move), not full game play. This dramatically simplifies the environment—each "episode" is a single move prediction, not a 40-move game—while still teaching all core GRPO concepts. Full game play can be positioned as a "going further" extension.

### Recommended notebook structure

**Section 1 — Setup and intuition (10 min).** Install dependencies, load Qwen 2.5 7B, introduce GRPO conceptually. Key teaching point: GRPO generates multiple candidate answers, scores them, and updates the policy to favor high-reward outputs. No critic network needed (unlike PPO), making it simpler to explain and implement.

**Section 2 — Building the chess environment (15 min).** Interactive python-chess demo: display boards, parse FEN, validate moves. Build the reward function component by component, testing each step. This is the most educational section—students see exactly how RL reward signals work.

**Section 3 — SFT warmstart (10 min, or load pre-trained checkpoint).** Brief supervised fine-tuning demonstration. Key teaching point: RL optimizes behavior the model can already approximately do; it can't teach entirely new skills. Show legal move rate jumping from ~5% to >90%.

**Section 4 — GRPO training (30–60 min).** Configure and launch `GRPOTrainer`. Watch wandb metrics in real-time. Key teaching points: group advantage estimation, why KL penalty prevents reward hacking, how clipping stabilizes training. Run 200–500 steps to show clear improvement.

**Section 5 — Evaluation and play (15 min).** Compare before/after metrics. Play interactive games against the model. Compare against GPT-4o on a few positions. Discuss limitations (the ~25–30% puzzle accuracy ceiling from Chess-R1) and what would be needed to go further.

### What should be pre-built vs. built live

Pre-compute the SFT warmstart checkpoint (saves 1–2 hours), the curated dataset of FEN positions with legal moves and Stockfish evaluations, and have Stockfish binary pre-installed in the Modal image. Build the reward function, GRPO configuration, and training loop live—these are the core educational content.

### Setting realistic expectations

The Chess-R1 paper's finding that **all models plateau at 25–30% puzzle accuracy** is actually a feature, not a bug, for an educational demo. It makes the project honest about RL's capabilities and limitations, provides a natural discussion point about why SFT data quality matters, and frames the broader question of what pretrained knowledge RL can and cannot amplify. The model will clearly learn (legal move rate increasing, move quality improving, rewards climbing)—and that visible learning signal is the entire point of the demo.

---

## Conclusion

The most important insight from this research is that **the entire stack for this demo already exists and has been validated**. Xiangqi-R1 proved the SFT→GRPO pipeline works for Qwen 7B on board games. TRL's GRPOTrainer provides a 10-line Python training loop with vLLM integration. Modal offers an official GRPO example with A100 GPUs at $2.50/hour. The `rezabonyadi/chess-language` repo provides a near-complete single-notebook reference implementation.

The non-obvious finding is that **chess is the right choice over checkers** despite being more complex—the ecosystem advantage (Stockfish for dense rewards, Lichess for data, python-chess for validation, 6+ reference projects) overwhelms the complexity disadvantage. And **SFT warmstart is the single most important design decision**: without it, GRPO cannot overcome the base model's chess ignorance, but with even a modest warmstart (5K–20K examples), RL reliably improves move quality and teaches the model to reason about positions.

The realistic scope for a demo on a $20 budget: a 7B model that goes from ~5% legal moves to >95%, shows measurable improvement in Stockfish evaluation scores, and can hold its own against GPT-4o on chess puzzles—all while teaching the student every core concept in modern LLM reinforcement learning.