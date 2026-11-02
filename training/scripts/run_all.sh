#!/usr/bin/env bash
# Full Review-2 pipeline on the pod, everything logged to W&B: raw-Qwen baseline -> QLoRA training -> fine-tuned generation.
# Launch detached (the SSH proxy kills session children): setsid nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
set -uo pipefail
cd /workspace
. /workspace/.wandb_env
export HF_HOME=/workspace/hf
export PATH=/workspace/venv-gen/bin:$PATH  # vLLM's compile step shells out to `ninja`
RUN=${RUN:-qlora_r16_e1}
mkdir -p preds logs

step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }

# A failed baseline must not block training; training failure does block the fine-tuned generation.
[ -n "${SKIP_BASELINE:-}" ] || step baseline venv-gen/bin/python scripts/generate.py --data data --split test --out preds/base_test.jsonl
step train venv-train/bin/python scripts/train_qlora.py --data data --out "runs/$RUN" &&
  step finetuned venv-gen/bin/python scripts/generate.py --data data --split test --adapter "runs/$RUN/adapter" \
    --adapter-artifact "adapter-$RUN:best" --out preds/ft_test.jsonl
echo "PIPELINE_DONE $(date -u +%T)"
