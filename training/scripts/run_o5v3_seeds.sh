#!/usr/bin/env bash
# O5 v3 extra seeds, local GPU: same data and settings as qlora_r16_o5v3 (seed 3407), then greedy test outputs.
#   bash training/scripts/run_o5v3_seeds.sh 1 2      (logs: output/o5v3/logs/pipeline.log, STEP_START / STEP_END markers)
# The guard (generate_local.py --guard-from) is run afterwards, so a checker change never needs a retrain.
export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/_netrc) PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=.venv-train/Scripts/python.exe; O=output/o5v3; G="--wandb-group review3-o5v3"
step() { echo "STEP_START $1 $(date +%T)" >> $O/logs/pipeline.log; "${@:2}" > $O/logs/$1.log 2>&1; local rc=$?
         echo "STEP_END $1 rc=$rc $(date +%T)" >> $O/logs/pipeline.log; return $rc; }
for s in "$@"; do
  R=$O/runs/qlora_r16_o5v3_s$s
  [ -d $R/adapter ] || step o5v3_s${s}_train $PY training/scripts/train_qlora.py --data data/datasets/o5_explanation --tasks explanation \
      --out $R --max-len 2048 --epochs 2 --eval-steps 25 --seed $s $G || continue
  step o5v3_s${s}_adapter $PY training/scripts/generate_local.py --prompts data/datasets/o5_explanation/test_prompts.jsonl \
      --adapter $R/adapter --out $O/preds/o5v3_s${s}_adapter_test.jsonl --batch 4 \
      --adapter-artifact itsbim-fpt-university/capstone_project/adapter-qlora_r16_o5v3_s$s:latest $G
done
echo "SEEDS_DONE $(date +%T)" >> $O/logs/pipeline.log
