#!/usr/bin/env bash
# After run_task3_live.sh: the V4 retry ladder went up to 1.3 (retry_v4.sh); the seed runs stopped at 1.2. Generate the test CVs at 1.3 for both seeds.
set -uo pipefail
cd /root/work
until grep -q PIPELINE_DONE logs/run_task3.log; do sleep 20; done
set -a; . /root/.wandb_env; set +a
export HF_HOME=/root/hf PATH=/root/venv-gen/bin:$PATH
step() { echo "STEP_START $1 $(date -u +%T)"; "${@:2}" > "logs/$1.log" 2>&1; local rc=$?; echo "STEP_END $1 rc=$rc $(date -u +%T)"; return $rc; }
python3 -c "import json; print(' '.join(json.loads(l)['id'] for l in open('data_v4/cv_extraction/test.jsonl')))" > logs/cv_test_ids.txt
for seed in 1 2; do
  step "v4_seed${seed}_retry13" /root/venv-gen/bin/python scripts/generate.py --data data_v4 --split test --repetition-penalty 1.3 \
    --ids-file logs/cv_test_ids.txt --adapter "runs/qlora_r16_v4gs_s${seed}/adapter" --out "preds/v4_s${seed}_retry13_test.jsonl" \
    --adapter-artifact "itsbim-fpt-university/capstone_project/adapter-qlora_r16_v4gs_s${seed}:latest" --wandb-group review3-task3
done
echo "RETRY_DONE $(date -u +%T)"
