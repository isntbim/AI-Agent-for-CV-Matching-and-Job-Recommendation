#!/usr/bin/env bash
# Drive the Task 3 pod from this PC over SSH (key ~/.ssh/runpod_ed25519). One script, so the GPU never waits on anyone:
#   bash training/scripts/task3_pod.sh pack                 # build the upload bundle locally (no pod needed)
#   bash training/scripts/task3_pod.sh up   <ip> <port> ["1 2"]   # upload, start run_task3.sh detached (extra seeds, default "1")
#   bash training/scripts/task3_pod.sh poll <ip> <port>     # step markers so far (STEP_START/STEP_END/PIPELINE_DONE)
#   bash training/scripts/task3_pod.sh get  <ip> <port>     # download preds, logs, adapters, run configs to output/task3/
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=output/task3
BUNDLE="$OUT/bundle.tgz"
KEY=~/.ssh/runpod_ed25519
ssh_() { ssh -i "$KEY" -p "$2" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "root@$1" "${@:3}"; }

case "$1" in
  pack)
    mkdir -p "$OUT/stage/scripts" "$OUT/stage/data_v4" "$OUT/stage/data_o5"
    cp training/scripts/{run_task3.sh,train_qlora.py,generate.py} "$OUT/stage/scripts/"
    cp -r data/datasets/qwen_training_v4/{cv_extraction,jd_extraction} "$OUT/stage/data_v4/"
    cp -r data/datasets/o5_explanation/{explanation,test_prompts.jsonl} "$OUT/stage/data_o5/"
    tar -czf "$BUNDLE" -C "$OUT/stage" scripts data_v4 data_o5
    rm -rf "$OUT/stage"
    tar -tzf "$BUNDLE" | sort; ls -la "$BUNDLE" ;;
  up)
    ssh_ "$2" "$3" "mkdir -p /root/work"
    ssh_ "$2" "$3" "tar --no-same-owner -xzf - -C /root/work" < "$BUNDLE"
    ssh_ "$2" "$3" "cd /root/work && mkdir -p logs && (SEEDS='${4-1}' setsid nohup bash scripts/run_task3.sh > logs/run_task3.log 2>&1 < /dev/null &) ; sleep 2; head -3 logs/run_task3.log; true" ;;
  poll)
    ssh_ "$2" "$3" "grep -h 'STEP_\|PIPELINE_DONE' /root/work/logs/run_task3.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; true" ;;
  get)
    mkdir -p "$OUT"
    ssh_ "$2" "$3" "cd /root/work && tar -czf - preds logs runs/*/adapter runs/*/run_config.json runs/*/log_history.json 2>/dev/null; true" \
      | tar -xzf - -C "$OUT"
    find "$OUT" -maxdepth 3 -newer "$BUNDLE" -type f | sort ;;
  *) echo "usage: $0 pack | up <ip> <port> | poll <ip> <port> | get <ip> <port>"; exit 2 ;;
esac
