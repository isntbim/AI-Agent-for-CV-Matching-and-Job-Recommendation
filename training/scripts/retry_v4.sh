#!/usr/bin/env bash
# Validate-and-retry, model half: regenerate the V4 test outputs that stayed invalid after scripts/qwen_repair.py
# (loops / broken JSON) at rising repetition penalties. Self-contained on a fresh pod: setup -> adapter -> generate,
# with no step waiting on the operator. Needs /workspace/.wandb_env, data_v4/{cv,jd}_extraction/test.jsonl, scripts/.
# Launch detached: setsid nohup bash scripts/retry_v4.sh > logs/retry_v4.log 2>&1 < /dev/null &
set -uo pipefail
cd /workspace
trap 'echo "RETRY_EXIT rc=$? $(date -u +%T)"' EXIT
set -e
export HF_HOME=/workspace/hf
mkdir -p logs preds runs
pip install -q --break-system-packages uv huggingface_hub
hf download Qwen/Qwen2.5-7B-Instruct > /dev/null &
download=$!
[ -d venv-gen ] || { uv venv venv-gen && VIRTUAL_ENV=venv-gen uv pip install -q vllm; }
VIRTUAL_ENV=venv-gen uv pip install -q wandb ninja
wait "$download"
echo "SETUP_DONE $(date -u +%T)"
. ./.wandb_env
venv-gen/bin/python -c "import wandb; wandb.Api().artifact('itsbim-fpt-university/capstone_project/adapter-qlora_r16_v4gs:v0').download('runs/qlora_r16_v4gs')"
PATH=/workspace/venv-gen/bin:$PATH venv-gen/bin/python scripts/generate.py --data data_v4 --split test \
  --adapter runs/qlora_r16_v4gs/adapter --adapter-artifact adapter-qlora_r16_v4gs:v0 \
  --ids "${RETRY_IDS:-cv_007,cv_016,cv_058,jd_233}" --repetition-penalty 1.1 1.15 1.2 1.3 --out preds/v4_retry_test.jsonl
echo "RETRY_DONE $(date -u +%T)"
