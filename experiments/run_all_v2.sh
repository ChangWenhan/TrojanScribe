#!/bin/bash
# run_all_v2.sh — orchestrator for the full 2026-09-08 v2 rerun.
# Sequential: main table -> ablations (dose / style diversity / trigger /
# retrieval window) -> cross-model -> baselines ->
# unified eval + summary. Every stage driver is idempotent (skips done runs),
# so a crash can be resumed by re-running this script.
#
# Usage: setsid bash experiments/run_all_v2.sh > results/logs/all_v2_driver.log 2>&1 &
set -u
ROOT=/mnt/disk/cwh/AgenticRAG
PY=/home/cwh/anaconda3/envs/agents/bin/python
LOG=$ROOT/results/logs
mkdir -p "$LOG"
STATUS=$ROOT/results/all_v2_status.txt
: > "$STATUS"

echo "=== $(date '+%F %T') v2 full rerun started ===" >> "$STATUS"

echo "--- stage 1: main table ---" >> "$STATUS"
bash "$ROOT/experiments/run_main_table_v2.sh" > "$LOG/main_table_v2_driver.log" 2>&1
echo "--- stage 2: ablations (dose / style diversity / trigger / window) ---" >> "$STATUS"
bash "$ROOT/experiments/run_ablation_v2.sh" > "$LOG/ablation_v2_driver.log" 2>&1
echo "--- stage 3: cross-model ---" >> "$STATUS"
bash "$ROOT/experiments/run_cross_model.sh" > "$LOG/cross_model_driver.log" 2>&1
echo "--- stage 4: react baselines ---" >> "$STATUS"
bash "$ROOT/experiments/run_baselines.sh" > "$LOG/baselines_driver.log" 2>&1
echo "--- stage 5: unified eval + summary ---" >> "$STATUS"
( cd "$ROOT/experiments" && "$PY" 13_unified_eval.py > "$LOG/13_unified_eval.log" 2>&1 )
( cd "$ROOT/experiments" && "$PY" summarize_ablation.py > "$LOG/summarize_ablation.log" 2>&1 )

echo "=== $(date '+%F %T') v2 full rerun done ===" >> "$STATUS"
cat "$STATUS"