"""Modal app for RL Chess — SFT → GRPO → Eval → Serve pipeline on serverless GPUs."""

import os
import subprocess
import sys

import modal

app = modal.App("rl-chess")

# --- Shared resources ---

vol = modal.Volume.from_name("rl-chess", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("make", "g++", "curl")
    .run_commands(
        "curl -sL https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_17.tar.gz | tar xzf - -C /tmp",
        "cd /tmp/Stockfish-sf_17/src && make -j$(nproc) build ARCH=x86-64-avx2",
        "cp /tmp/Stockfish-sf_17/src/stockfish /usr/local/bin/stockfish",
        "rm -rf /tmp/Stockfish-sf_17",
    )
    .pip_install(
        "torch",
        "trl>=0.29.0",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "python-chess",
        "wandb",
        "vllm",
        "openai",
    )
    .env(
        {
            "USER": "modal",
            "STOCKFISH_PATH": "/usr/local/bin/stockfish",
        }
    )
    .add_local_dir("scripts", remote_path="/root/scripts")
)

VOL_MOUNT = "/scratch/modal"
CHECKPOINTS = f"{VOL_MOUNT}/.rv/checkpoints"
SFT_MERGED = f"{CHECKPOINTS}/chess-sft/merged"
GRPO_MERGED = f"{CHECKPOINTS}/chess-grpo/merged"

secrets = [modal.Secret.from_name("rl-chess-secrets")]


# --- Functions ---


@app.function(
    image=image,
    volumes={VOL_MOUNT: vol},
    secrets=secrets,
    timeout=1800,
)
def data_prep():
    subprocess.run(
        [sys.executable, "/root/scripts/data_prep.py"],
        check=True,
    )
    vol.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    secrets=secrets,
    timeout=10800,
)
def sft_train():
    vol.reload()
    subprocess.run(
        [sys.executable, "/root/scripts/sft_train.py"],
        check=True,
    )
    vol.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    secrets=secrets,
    timeout=10800,
)
def grpo_train():
    # vLLM colocate mode needs distributed env vars
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
        }
    )
    vol.reload()
    subprocess.run(
        [
            sys.executable,
            "/root/scripts/grpo_train.py",
            "--model_path",
            SFT_MERGED,
        ],
        check=True,
    )
    vol.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    secrets=secrets,
    timeout=3600,
)
def evaluate():
    vol.reload()
    subprocess.run(
        [
            sys.executable,
            "/root/scripts/evaluate.py",
            "--num_puzzles",
            "100",
            "--num_games",
            "10",
        ],
        check=True,
    )
    vol.commit()


@app.function(
    image=image,
    volumes={VOL_MOUNT: vol},
    timeout=60,
)
def get_eval_results() -> dict:
    import json

    vol.reload()
    results_path = f"{VOL_MOUNT}/.rv/outputs/chess-eval/eval_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {"error": "No eval results found yet"}


# --- Serving ---

VLLM_PORT = 8000


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    scaledown_window=15 * 60,
    timeout=10 * 60,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    vol.reload()
    model_path = GRPO_MERGED if os.path.exists(GRPO_MERGED) else SFT_MERGED
    print(f"Serving model: {model_path}")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.9",
        ]
    )


# --- Entrypoints ---


@app.local_entrypoint()
def run_data_prep():
    data_prep.remote()
    print("Data prep complete!")


@app.local_entrypoint()
def run_sft():
    sft_train.remote()
    print("SFT training complete!")


@app.local_entrypoint()
def run_grpo():
    grpo_train.remote()
    print("GRPO training complete!")


@app.local_entrypoint()
def run_eval():
    evaluate.remote()
    print("Evaluation complete!")


@app.local_entrypoint()
def fetch_results():
    import json

    results = get_eval_results.remote()
    print(json.dumps(results, indent=2))
