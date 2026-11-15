#!/usr/bin/env bash
# Task 3 pod run, one script so the GPU never waits on anyone (A100 SXM, container disk only):
#   setup -> O5 explanation adapter (train, then base vs adapter on the test prompts)
#         -> V4 recipe with two more seeds (train, then test generation at the retry penalties).
# Upload first: scripts/, data_v4/ (= data/datasets/qwen_training_v4), data_o5/ (= data/datasets/o5_explanation) to /root/work.
# Launch detached: cd /root/work && setsid nohup bash scripts/run_task3.sh > logs/run_task3.log 2>&1 < /dev/null & disown
# Progress markers: STEP_START / STEP_END <name> rc=<n>, PIPELINE_DONE at the end.
set -uo pipefail
cd /root/work
export HF_HOME=/root/hf PATH=/root/venv-gen/bin:$PATH  # venvs and weights on the container disk: /workspace can be slow network FS
mkdir -p preds logs runs
step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }

setup() {
  pip install -q --break-system-packages uv huggingface_hub
  hf download Qwen/Qwen2.5-7B-Instruct > /dev/null &
  local download=$!
  [ -d /root/venv-train ] || { uv venv /root/venv-train && VIRTUAL_ENV=/root/venv-train uv pip install unsloth==2026.10.9; }
  [ -d /root/venv-gen ] || { uv venv /root/venv-gen && VIRTUAL_ENV=/root/venv-gen uv pip install vllm==0.32.0; }
  VIRTUAL_ENV=/root/venv-train uv pip install -q wandb; VIRTUAL_ENV=/root/venv-gen uv pip install -q wandb
  VIRTUAL_ENV=/root/venv-gen uv pip install -q ninja
  /root/venv-train/bin/python -c "import unsloth, trl, torch; print('train ok', torch.cuda.get_device_name(0))"
  /root/venv-gen/bin/python -c "import vllm; print('gen ok', vllm.__version__)"
  wait "$download" && echo "model ok"
}
TRAIN=/root/venv-train/bin/python
GEN=/root/venv-gen/bin/python

step setup setup || { echo "PIPELINE_DONE setup_failed $(date -u +%T)"; exit 1; }

# O5: short examples (at most ~1.4k tokens with the Vietnamese rows, checked locally), so a 2k context and 2 epochs
step o5_train $TRAIN scripts/train_qlora.py --data data_o5 --tasks explanation --out runs/qlora_r16_o5 \
  --max-len 2048 --epochs 2 --eval-steps 25 --no-wandb
step o5_base $GEN scripts/generate.py --prompts data_o5/test_prompts.jsonl --out preds/o5_base_test.jsonl \
  --max-tokens 700 --no-wandb
[ -d runs/qlora_r16_o5/adapter ] && step o5_adapter $GEN scripts/generate.py --prompts data_o5/test_prompts.jsonl \
  --adapter runs/qlora_r16_o5/adapter --out preds/o5_adapter_test.jsonl --max-tokens 700 --no-wandb

# Seeds: the V4 recipe (run_v4.sh) with extra seeds; V4 itself used the default 3407
for seed in ${SEEDS-1}; do  # SEEDS="" none, "1" one extra (default), "1 2" two
  step "v4_seed${seed}_train" $TRAIN scripts/train_qlora.py --data data_v4 --out "runs/qlora_r16_v4gs_s${seed}" \
    --upsample-source gold:4 --seed "$seed" --no-wandb &&
  step "v4_seed${seed}_gen" $GEN scripts/generate.py --data data_v4 --split test --repetition-penalty 1.05 1.1 1.15 1.2 \
    --adapter "runs/qlora_r16_v4gs_s${seed}/adapter" --out "preds/v4_s${seed}_test.jsonl" --no-wandb
done
echo "PIPELINE_DONE $(date -u +%T)"
