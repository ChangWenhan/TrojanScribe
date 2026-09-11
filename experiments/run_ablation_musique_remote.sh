#!/bin/bash
# run_ablation_musique_remote.sh — MuSiQue ablation arms for a victim served on
# a REMOTE vLLM endpoint (192.168.31.141). Code + KB stay local; the victim's
# poison generation (inject) AND eval all go through AGENTIC_RAG_BASE_URL.
#
# Idempotent: an arm result with an 'after' section is skipped.
# Usage: bash experiments/run_ablation_musique_remote.sh <victim> <status_file>
set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
FROZEN_MUS=$ROOT/research/frozen/musique_targets_59.json
KB_COLL=musique_kb
REMOTE_URL=http://192.168.31.141:8000/v1
REMOTE_USER=hp
REMOTE_HOST=192.168.31.141
mkdir -p "$LOGDIR"

MODEL="${1:?usage: run_ablation_musique_remote.sh <victim> <status_file>}"; shift
STATUS="${1:?}"; shift
: > "$STATUS"
KB_DIR=${MUSIQUE_KB_DIR:-$ROOT/data/chroma_musique_141}
[ -d "$KB_DIR" ] || { echo "MUSIQUE KB DIR missing: $KB_DIR — copy it first (cp -r data/chroma_musique data/chroma_musique_141) or set MUSIQUE_KB_DIR" >&2; exit 1; }
export AGENTIC_RAG_BASE_URL="$REMOTE_URL"
case "$MODEL" in
  qwen3-8b) export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}' ;;
  *)        unset AGENTIC_RAG_CHAT_KWARGS ;;
esac

run_arm () {  # $1 arm  $2... 08 args
  local arm=$1; shift
  local out="$ROOT/results/08_longtail_${MODEL}_${arm}_musique.json"
  if [ -s "$out" ] && "$PY" -c "import json,sys
try:
    d=json.load(open('$out')); v=d.get('variants',{})
    sys.exit(0 if any('after' in x for x in v.values()) else 1)
except Exception:
    sys.exit(1)" 2>/dev/null; then
    echo "SKIP (done): $out" >> "$STATUS"
    return
  fi
  local mname="08_longtail_${MODEL}_musique"
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="${MODEL}_${arm}_mus_remote_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  export AGENTIC_RAG_RUN_ID="$run_id"
  echo "=== $(date '+%F %T') arm $MODEL/$arm (musique remote) ===" | tee -a "$STATUS"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py "$@" \
      --model "$MODEL" \
      --target-records "$FROZEN_MUS" \
      --kb-dir "$KB_DIR" --kb-collection "$KB_COLL" \
      --clean-from "$ROOT/results/$mname.json" \
      --results-name "08_longtail_${MODEL}_${arm}_musique" \
      > "$rundir/08_${MODEL}_${arm}_musique.log" 2>&1 ) \
    && echo "$MODEL/$arm musique OK ($run_id)" >> "$STATUS" || echo "$MODEL/$arm musique RUN_FAILED ($run_id)" >> "$STATUS"
  unset AGENTIC_RAG_RUN_ID
}

# dose
run_arm vol2  --volume 2 --variants cluster
run_arm vol4  --volume 4 --variants cluster
run_arm vol6  --volume 6 --variants cluster
# style diversity
run_arm embed_hybrid --volume 8 --variants embed_hybrid
run_arm mono         --volume 8 --variants cluster_mono
run_arm nodiv        --volume 8 --variants cluster_nodiv
run_arm greedy       --volume 8 --variants cluster_greedy
# trigger
run_arm semantic     --volume 8 --variants cluster --trigger-kind semantic
run_arm trig_always  --volume 8 --variants cluster --trigger-kind always --always-probability 1.0
# retrieval window
run_arm topk4        --volume 8 --variants cluster --top-k 4
run_arm topk16       --volume 8 --variants cluster --top-k 16

echo "=== $MODEL musique ablation (remote) done ===" >> "$STATUS"