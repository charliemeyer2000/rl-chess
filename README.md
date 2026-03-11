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

What we're attempting to do here is teach some small language model, in this case, `Qwen-2.5-7B-Instruct` to beat ChatGPT 4o in a game of chess. Now, let's think through why this is a hard task, and what we need to do to get this done. 


### Task Overview

Most language models aren't particularly good at playing chess - they'll hallucinate, make illegal moves, or just don't know enough chess strategy to be able to win. Moreover, they also aren't great a following instructions, and we need to specifically teach them our format of how we'll play a game of chess with them. 

There are various steps 

### Strategy

1. We need to download, tune, and shape our data to ensure that we are training our model on the correct format, and with a high-quality dataset of games. 
2. We will do supervised fine-tuning (SFT) to teach our language model how we want to play chess, and get it to understand good chess games
3. We will do reinforcement learning (RL) to have our language model play against 4o, give it feedback as it plays, until it can win!
    - We will use [stockfish](https://stockfishchess.org/) as our reward function - we can give our agent a score based on their performance vs. stockfish as our "judge".
4. We will make a quick simulation site so you can play chess against the language model you just trained!

### Demo Code

In this codebase, there are three files:
- `modal-run.ipynb` - example of this with modal
- `rivanna-run.ipynb` - example of this with UVA HPC (using `rv` cli). 
- `cluster.ipynb` - example of this on a regular node (assuming some GPU-accelerated node, detects cuda/mps). 