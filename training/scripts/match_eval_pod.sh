#!/usr/bin/env bash
# Validation-pair extraction on the A100: V4 (bf16 base + adapter, vLLM) over every CV/JD prompt of
# scripts/match_eval.py --export-prompts, then a retry pass for the documents the local repair could not fix.
# Self-contained on a fresh pod (container disk, no network volume): setup -> pass 1 -> wait for retry ids -> pass 2.
# Needs in /root/job (container disk; /workspace can be a slow network FS): prompts.jsonl, adapter/ (output/review2/qwen/v4/adapter), scripts/generate.py.
# Launch detached: setsid nohup bash scripts/match_eval_pod.sh > logs/match.log 2>&1 < /dev/null &
# Operator, after PASS1_DONE: download preds/match_rp105.jsonl, run match_eval --replay, upload retry_ids.txt
# (an empty file = nothing to retry). Without it the pod stops waiting after 40 minutes.
set -uo pipefail
cd /root/job
trap 'echo "POD_EXIT rc=$? $(date -u +%T)"' EXIT
set -e
export HF_HOME=/root/hf
mkdir -p logs preds
pip install -q --break-system-packages uv huggingface_hub
hf download Qwen/Qwen2.5-7B-Instruct > /dev/null &
download=$!
uv venv venv-gen && VIRTUAL_ENV=venv-gen uv pip install -q vllm
wait "$download"
echo "SETUP_DONE $(date -u +%T)"
GEN="venv-gen/bin/python scripts/generate.py --prompts prompts.jsonl --adapter adapter --no-wandb"
PATH=/root/job/venv-gen/bin:$PATH $GEN --repetition-penalty 1.05 --out preds/match_rp105.jsonl
echo "PASS1_DONE $(date -u +%T)"
for _ in $(seq 240); do [ -f retry_ids.txt ] && break; sleep 10; done
[ -f retry_ids.txt ] || { echo "NO_RETRY_IDS $(date -u +%T)"; exit 0; }
if [ -s retry_ids.txt ]; then
  PATH=/root/job/venv-gen/bin:$PATH $GEN --ids-file retry_ids.txt --repetition-penalty 1.1 1.15 1.2 --out preds/match_retry.jsonl
fi
echo "PASS2_DONE $(date -u +%T)"
