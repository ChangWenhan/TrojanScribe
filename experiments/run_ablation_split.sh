#!/bin/bash
# run_ablation_split.sh — run an EXPLICIT subset of gpt-oss-20b ablation arms
# against one vLLM endpoint with one dedicated KB dir (two instances of this
# script with disjoint arms + disjoint KB dirs can run in parallel).
# Usage: bash experiments/run_ablation_split.sh <base_url> <kb_dir> <status_file> <arm>...
set -u
ROOT=/mnt/disk/cwh/AgenticRAG
PY=/home/cwh/anaconda3/envs/agents/bin/python
LOGDIR=$ROOT/results/logs
mkdir -p "$LOGDIR"
BASE_URL="${1:?usage: run_ablation_split.sh <base_url> <kb_dir> <status_file> <arm>...}"
KB_DIR="${2:?}"
STATUS="${3:?}"
shift 3
[ $# -gt 0 ] || { echo "no arms given" >> "$STATUS"; exit 1; }
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8

model="gpt-oss-20b"
clean_src="$ROOT/results/08_longtail_${model}.json"
[ -f "$clean_src" ] || { echo "clean baseline $clean_src missing" >> "$STATUS"; exit 1; }
curl -s --max-time 5 "$BASE_URL/models" | grep -q "\"id\".*\"$model\"" \
  || { echo "SERVER_MISSING: $BASE_URL (model $model)" >> "$STATUS"; exit 1; }

flags_for () {
  case "$1" in
    vol2)        echo "--volume 2" ;;
    vol4)        echo "--volume 4" ;;
    vol6)        echo "--volume 6" ;;
    embed_hybrid) echo "--variants embed_hybrid" ;;
    mono)        echo "--variants cluster_mono" ;;
    nodiv)       echo "--variants cluster_nodiv" ;;
    greedy)      echo "--variants cluster_greedy" ;;
    semantic)    echo "--trigger-kind semantic" ;;
    trig_always) echo "--trigger-kind always --always-probability 1.0" ;;
    topk4)       echo "--top-k 4" ;;
    topk16)      echo "--top-k 16" ;;
    *)           return 1 ;;
  esac
}

run_arm () {  # $1 arm, rest = extra flags
  local arm="$1"; shift
  local out="$ROOT/results/08_longtail_${model}_${arm}.json"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$out" ] && "$PY" - "$out" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
vs = d.get("variants") or {}
if any(v.get("after") for v in vs.values()):
    sys.exit(0)
sys.exit(1)
EOF
  then
    echo "SKIP (done): $model/$arm" >> "$STATUS"
    return
  fi
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="${model}_${arm}_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  export AGENTIC_RAG_RUN_ID="$run_id" AGENTIC_RAG_BASE_URL="$BASE_URL"
  echo "=== $(date '+%F %T') arm $model/$arm ($BASE_URL, kb $(basename "$KB_DIR")) ===" | tee -a "$STATUS"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
      --targets 60 --volume 8 --use-shared-targets \
      --model "$model" \
      --kb-dir "$KB_DIR" \
      --clean-from "$clean_src" \
      --results-name "08_longtail_${model}_${arm}" \
      $(flags_for "$arm") "$@" \
      > "$rundir/08_${model}_${arm}.log" 2>&1 ) \
    && echo "$model/$arm OK ($run_id)" >> "$STATUS" || echo "$model/$arm RUN_FAILED ($run_id)" >> "$STATUS"
}

for arm in "$@"; do run_arm "$arm"; done
echo "=== split instance done $(date '+%F %T') ===" >> "$STATUS"
