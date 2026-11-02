#!/usr/bin/env bash
# One-time pod setup. Unsloth and vLLM pin different torch builds, so each gets its own venv.
# Expects the network volume at /workspace and the SFT data copied to /workspace/data.
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/hf  # keep model downloads on the volume
pip install -q --break-system-packages uv huggingface_hub  # Ubuntu 24.04 image: throwaway container, system pip is fine
hf download Qwen/Qwen2.5-7B-Instruct > /dev/null &  # overlaps with the installs below
download=$!
[ -d venv-train ] || { uv venv venv-train && VIRTUAL_ENV=venv-train uv pip install unsloth; }
VIRTUAL_ENV=venv-train uv pip install -q wandb
[ -d venv-gen ] || { uv venv venv-gen && VIRTUAL_ENV=venv-gen uv pip install vllm; }  # create before the extras below
VIRTUAL_ENV=venv-gen uv pip install -q wandb ninja
venv-train/bin/python -c "import unsloth, trl, torch; print('train ok', trl.__version__, torch.__version__, torch.cuda.get_device_name(0))"
venv-gen/bin/python -c "import vllm; print('gen ok', vllm.__version__)"
wait "$download" && echo "model ok"
