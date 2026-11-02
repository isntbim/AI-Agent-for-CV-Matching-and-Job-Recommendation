#!/usr/bin/env bash
# V4: V3b's recipe (all CV labels, gold x4) on the gold-style-filtered teacher labels (GOLD_STYLE_FILTER.md).
# Data: local data/datasets/qwen_training_v4/{cv,jd}_extraction copied to /workspace/data_v4 (V3 data stays in data/).
# Launch detached: setsid nohup bash scripts/run_v4.sh > logs/run_v4.log 2>&1 < /dev/null & disown
set -uo pipefail
cd /workspace
. /workspace/.wandb_env
export HF_HOME=/workspace/hf PATH=/workspace/venv-gen/bin:$PATH  # vLLM's compile step shells out to `ninja`
mkdir -p preds logs
step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }

step train_v4 venv-train/bin/python scripts/train_qlora.py --data data_v4 --out runs/qlora_r16_v4gs --upsample-source gold:4 &&
  step v4_rp venv-gen/bin/python scripts/generate.py --data data_v4 --split test --repetition-penalty 1.05 \
    --adapter runs/qlora_r16_v4gs/adapter --adapter-artifact adapter-qlora_r16_v4gs:best --out preds/v4_rp_test.jsonl
echo "PIPELINE_DONE $(date -u +%T)"
