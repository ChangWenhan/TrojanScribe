#!/bin/bash
# HotpotQA gpt-oss redo (A4 fix) — 141 share of the dual-machine split:
#   waits for the local main-row flag, serves gpt-oss on 141, runs 6 arms on an
#   isolated hotpot KB clone, then the gpt-oss->qwen3 and gpt-oss->llama
#   cross-model pairs, then restores qwen3-8b on 141.
set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/cross_model_gptoss_141_status.txt
FLAG=$ROOT/results/hotpot_gptoss_main_done.flag
REMOTE_HOST=hp@192.168.31.141
REMOTE_URL=http://192.168.31.141:8000/v1
EFFORT='{"reasoning_effort": "low"}'
mkdir -p "$LOGDIR"

# KB clones used by this one-off redo were removed in the 2026-09-11 cleanup;
# recreate before rerunning (cp -r data/chroma data/chroma_gpt-oss-20b_b;
# cp -r data/chroma data/chroma_xsm141).
for d in "$ROOT/data/chroma_gpt-oss-20b_b" "$ROOT/data/chroma_xsm141"; do
  [ -d "$d" ] || { echo "KB clone missing: $d — see comment in this script" >&2; exit 1; }
done

echo "=== $(date '+%F %T') 141 hotpot gpt-oss redo armed; waiting for the main-row flag ===" >> "$STATUS"
while [ ! -f "$FLAG" ]; do sleep 20; done
sleep 15

serve_remote () {  # $1 model
  ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash -s $1" \
      < "$ROOT/scripts/serve_141.sh" > "$LOGDIR/serve141_$1.log" 2>&1
  for _ in $(seq 1 90); do
    curl -s --max-time 3 "$REMOTE_URL/models" 2>/dev/null | grep -q "\"id\": *\"$1\"" && return 0
    sleep 5
  done
  return 1
}

export AGENTIC_RAG_TOP_LEVEL_KWARGS="$EFFORT"
export AGENTIC_RAG_PAYLOAD_MAX_TOKENS=768
unset AGENTIC_RAG_CHAT_KWARGS
export AGENTIC_RAG_BASE_URL="$REMOTE_URL"

serve_remote gpt-oss-20b || { echo "=== gpt-oss-20b SERVE FAILED on 141 ===" >> "$STATUS"; exit 1; }
echo "=== $(date '+%F %T') 141 serving gpt-oss-20b; running 6 arms ===" >> "$STATUS"
"$PY" "$ROOT/experiments/harness_smoke.py" gpt-oss-20b \
    >> "$LOGDIR/harnesssmoke_gpt-oss-20b_141.log" 2>&1 \
  || { echo "gpt-oss-20b HARNESS_SMOKE_FAILED on 141" >> "$STATUS"; exit 1; }

# 141 share of the 11 arms (isolated KB clone: data/chroma_gpt-oss-20b_b)
FORCE=1 bash "$ROOT/experiments/run_ablation_split.sh" "$REMOTE_URL" \
  "$ROOT/data/chroma_gpt-oss-20b_b" "$STATUS" \
  nodiv greedy semantic trig_always topk4 topk16

# cross-model pairs against remote victims (replay from the new main row)
serve_remote qwen3-8b \
  && FORCE=1 bash "$ROOT/experiments/run_cross_model_split.sh" remote "$REMOTE_URL" \
      "$ROOT/data/chroma_xsm141" "$STATUS" gpt-oss-20b:qwen3-8b
serve_remote llama-3.1-8b \
  && FORCE=1 bash "$ROOT/experiments/run_cross_model_split.sh" remote "$REMOTE_URL" \
      "$ROOT/data/chroma_xsm141" "$STATUS" gpt-oss-20b:llama-3.1-8b

serve_remote qwen3-8b || true
( cd "$ROOT/experiments" && "$PY" summarize_ablation.py ) >> "$STATUS" 2>&1 \
  && echo "=== summary refreshed ===" >> "$STATUS"
echo "=== $(date '+%F %T') 141 hotpot gpt-oss redo done; qwen3-8b restored ===" >> "$STATUS"
