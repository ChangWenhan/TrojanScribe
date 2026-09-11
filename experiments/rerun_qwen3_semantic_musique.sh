#!/bin/bash
# One-off repair (2026-09-10): the qwen3-8b MuSiQue semantic arm was killed
# mid-injection at 14:40 when the gpt-oss interception driver ran
# `pkill -f longtail_attack.py` — remote-victim arms execute their harness
# LOCALLY (only the victim server is on 141), so they matched the pattern.
# Waits for the remote driver to finish its queue, then reruns the arm.
set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
export AGENTIC_RAG_BASE_URL="http://192.168.31.141:8000/v1"
export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}'
PY=/home/cwh/anaconda3/envs/agents/bin/python
ROOT=/mnt/disk/cwh/AgenticRAG
STATUS=$ROOT/results/ablation_musique_qwen3_status.txt
FROZEN_MUS=$ROOT/research/frozen/musique_targets_59.json
KB_DIR=$ROOT/data/chroma_musique_141

echo "=== $(date '+%F %T') semantic rerun armed; waiting for remote driver ===" >> "$STATUS"
while pgrep -f "[r]un_ablation_musique_remote.sh" >/dev/null; do sleep 20; done

stamp=$(date +%Y%m%d_%H%M%S)
run_id="qwen3-8b_semantic_mus_rerun_${stamp}"
rundir="$ROOT/results/runs/$run_id"
mkdir -p "$rundir"
export AGENTIC_RAG_RUN_ID="$run_id"
echo "=== $(date '+%F %T') arm qwen3-8b/semantic (musique remote, rerun after collateral kill) ===" >> "$STATUS"
( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
    --volume 8 --variants cluster --trigger-kind semantic \
    --model qwen3-8b \
    --target-records "$FROZEN_MUS" \
    --kb-dir "$KB_DIR" --kb-collection musique_kb \
    --clean-from "$ROOT/results/08_longtail_qwen3-8b_musique.json" \
    --results-name "08_longtail_qwen3-8b_semantic_musique" \
    > "$rundir/08_qwen3-8b_semantic_musique.log" 2>&1 ) \
  && echo "qwen3-8b/semantic musique OK ($run_id)" >> "$STATUS" \
  || echo "qwen3-8b/semantic musique RUN_FAILED ($run_id)" >> "$STATUS"
unset AGENTIC_RAG_RUN_ID

( cd "$ROOT/experiments" && "$PY" summarize_ablation.py ) >> "$STATUS" 2>&1 \
  && echo "=== summary refreshed ===" >> "$STATUS"
tail -8 "$STATUS"
