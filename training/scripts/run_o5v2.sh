#!/usr/bin/env bash
# O5 v2 pod run (A100 SXM, container disk only): setup -> train the explanation adapter on the v2 plain-statement
# prompts -> base and adapter on the 244 v2 test prompts. Everything logs live to W&B (group review3-o5v2).
# Upload first: scripts/{run_o5v2.sh,train_qlora.py,generate.py}, data_o5/ (= data/datasets/o5_explanation) to /root/work,
# and WANDB_API_KEY=... in /root/.wandb_env. Launch detached:
#   cd /root/work && setsid nohup bash scripts/run_o5v2.sh > logs/run.log 2>&1 < /dev/null &
# Progress markers: STEP_START / STEP_END <name> rc=<n>, PIPELINE_DONE at the end.
set -uo pipefail
cd /root/work
set -a; . /root/.wandb_env; set +a
export HF_HOME=/root/hf PATH=/root/venv-gen/bin:$PATH
mkdir -p preds logs runs
step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }

setup() {
  pip install -q --break-system-packages uv huggingface_hub
  hf download Qwen/Qwen2.5-7B-Instruct > /dev/null &
  local download=$!
  [ -d /root/venv-train ] || { uv venv /root/venv-train && VIRTUAL_ENV=/root/venv-train uv pip install unsloth==2026.10.9 wandb; }
  [ -d /root/venv-gen ] || { uv venv /root/venv-gen && VIRTUAL_ENV=/root/venv-gen uv pip install vllm==0.32.0 wandb; }
  VIRTUAL_ENV=/root/venv-gen uv pip install -q ninja
  /root/venv-train/bin/python -c "import unsloth, trl, torch, wandb; print('train ok', torch.cuda.get_device_name(0))"
  /root/venv-gen/bin/python -c "import vllm, wandb; print('gen ok', vllm.__version__)"
  wait "$download" && echo "model ok"
}
TRAIN=/root/venv-train/bin/python
GEN=/root/venv-gen/bin/python
G="--wandb-group review3-o5v2"

step setup setup || { echo "PIPELINE_DONE setup_failed $(date -u +%T)"; exit 1; }
step o5v2_train $TRAIN scripts/train_qlora.py --data data_o5 --tasks explanation --out runs/qlora_r16_o5v2 \
  --max-len 2048 --epochs 2 --eval-steps 25 $G
step o5v2_base $GEN scripts/generate.py --prompts data_o5/test_prompts.jsonl --out preds/o5v2_base_test.jsonl --max-tokens 700 $G
[ -d runs/qlora_r16_o5v2/adapter ] && step o5v2_adapter $GEN scripts/generate.py --prompts data_o5/test_prompts.jsonl \
  --adapter runs/qlora_r16_o5v2/adapter --out preds/o5v2_adapter_test.jsonl --max-tokens 700 \
  --adapter-artifact "itsbim-fpt-university/capstone_project/adapter-qlora_r16_o5v2:latest" $G
echo "PIPELINE_DONE $(date -u +%T)"
