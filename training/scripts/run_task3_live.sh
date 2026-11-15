#!/usr/bin/env bash
# Rest of run_task3.sh with live W&B logging (setup and o5_train already done in the first run).
set -uo pipefail
cd /root/work
set -a; . /root/.wandb_env; set +a
export HF_HOME=/root/hf PATH=/root/venv-gen/bin:$PATH
mkdir -p preds logs runs
step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }
TRAIN=/root/venv-train/bin/python
GEN=/root/venv-gen/bin/python
G="--wandb-group review3-task3"
step o5_base $GEN scripts/generate.py --prompts data_o5/test_prompts.jsonl --out preds/o5_base_test.jsonl --max-tokens 700 $G
[ -d runs/qlora_r16_o5/adapter ] && step o5_adapter $GEN scripts/generate.py --prompts data_o5/test_prompts.jsonl \
  --adapter runs/qlora_r16_o5/adapter --out preds/o5_adapter_test.jsonl --max-tokens 700 $G
for seed in ${SEEDS-1 2}; do
  step "v4_seed${seed}_train" $TRAIN scripts/train_qlora.py --data data_v4 --out "runs/qlora_r16_v4gs_s${seed}" \
    --upsample-source gold:4 --seed "$seed" $G &&
  step "v4_seed${seed}_gen" $GEN scripts/generate.py --data data_v4 --split test --repetition-penalty 1.05 1.1 1.15 1.2 \
    --adapter "runs/qlora_r16_v4gs_s${seed}/adapter" --out "preds/v4_s${seed}_test.jsonl" \
    --adapter-artifact "itsbim-fpt-university/capstone_project/adapter-qlora_r16_v4gs_s${seed}:latest" $G
done
echo "PIPELINE_DONE $(date -u +%T)"
