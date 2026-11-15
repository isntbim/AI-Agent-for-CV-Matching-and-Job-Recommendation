#!/usr/bin/env bash
# O5 v3 reproducible test outputs: one prompt per batch (no padding, layout-independent), then the template guard.
#   bash training/scripts/run_o5v3_solo.sh      (logs: output/o5v3/logs/pipeline.log, STEP_START / STEP_END markers)
# Why: batched greedy outputs change with batch size and padding (2026-11-13 test: same layout twice 24/24 identical,
# 2 vs 4 copies of one prompt 9/24). generate_local.py sends a single prompt twice, which is the fixed layout used here.
export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/_netrc) PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=.venv-train/Scripts/python.exe; O=output/o5v3; G="--wandb-group review3-o5v3"
P="--prompts data/datasets/o5_explanation/test_prompts.jsonl"
step() { echo "STEP_START $1 $(date +%T)" >> $O/logs/pipeline.log; "${@:2}" > $O/logs/$1.log 2>&1; local rc=$?
         echo "STEP_END $1 rc=$rc $(date +%T)" >> $O/logs/pipeline.log; return $rc; }
for s in "" _s1 _s2; do
  R=$O/runs/qlora_r16_o5v3$s; A=$O/preds/o5v3${s}_adapter_test_solo.jsonl
  step o5v3${s}_solo $PY training/scripts/generate_local.py $P --adapter $R/adapter --out $A --batch 1 \
      --adapter-artifact itsbim-fpt-university/capstone_project/adapter-qlora_r16_o5v3$s:latest $G || continue
  step o5v3${s}_guard_solo $PY training/scripts/generate_local.py $P --adapter $R/adapter --guard-from $A \
      --out $O/preds/o5v3${s}_guard_test_solo.jsonl $G
done
echo "SOLO_DONE $(date +%T)" >> $O/logs/pipeline.log
