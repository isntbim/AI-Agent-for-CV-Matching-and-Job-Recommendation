#!/usr/bin/env bash
# V3b, queued behind run_v3.sh (V3a): keep ALL CV labels (incl. external teachers), upsample gold x4.
# A file-based wait, not a pgrep loop (a pgrep pattern matches its own waiter's command line).
# Launch detached: setsid nohup bash scripts/run_v3b.sh > logs/run_v3b.log 2>&1 < /dev/null & disown
set -uo pipefail
cd /workspace
until grep -q PIPELINE_DONE logs/run_v3.log 2>/dev/null; do sleep 30; done
. /workspace/.wandb_env
export HF_HOME=/workspace/hf PATH=/workspace/venv-gen/bin:$PATH
step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }

step train_v3b venv-train/bin/python scripts/train_qlora.py --data data --out runs/qlora_r16_v3all --upsample-source gold:4 &&
  step v3b_rp venv-gen/bin/python scripts/generate.py --data data --split test --repetition-penalty 1.05 \
    --adapter runs/qlora_r16_v3all/adapter --adapter-artifact adapter-qlora_r16_v3all:best --out preds/v3b_rp_test.jsonl
echo "PIPELINE_DONE $(date -u +%T)"
