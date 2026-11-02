#!/usr/bin/env bash
# Review-2 follow-up, all logged to W&B: re-generate baseline + V2 with a repetition penalty (fixes greedy loops),
# then train V3 on the gold-style mix (no external teacher labels, gold x4) and generate with the same penalty.
# Launch detached: setsid nohup bash scripts/run_v3.sh > logs/run_v3.log 2>&1 < /dev/null & disown
set -uo pipefail
cd /workspace
. /workspace/.wandb_env
export HF_HOME=/workspace/hf PATH=/workspace/venv-gen/bin:$PATH  # vLLM's compile step shells out to `ninja`
RP=1.05
mkdir -p preds logs

step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }
gen() { venv-gen/bin/python scripts/generate.py --data data --split test --repetition-penalty $RP "$@"; }

step base_rp gen --out preds/base_rp_test.jsonl
step v2_rp gen --adapter runs/qlora_r16_e1/adapter --adapter-artifact adapter-qlora_r16_e1:best --out preds/v2_rp_test.jsonl
step train_v3 venv-train/bin/python scripts/train_qlora.py --data data --out runs/qlora_r16_v3mix \
    --exclude-source external --upsample-source gold:4 &&
  step v3_rp gen --adapter runs/qlora_r16_v3mix/adapter --adapter-artifact adapter-qlora_r16_v3mix:best --out preds/v3_rp_test.jsonl
echo "PIPELINE_DONE $(date -u +%T)"
