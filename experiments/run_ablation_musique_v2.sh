#!/bin/bash
# Ablation driver v2 — MuSiQue (59 frozen targets), all 11 ablation settings on
# all four models. Mirrors run_ablation_v2.sh but for the musique KB:
#   poison dose        : vol2 / vol4 / vol6              (8 = main-table cluster row)
#   style diversity    : embed_hybrid / mono / nodiv / greedy (full = main table)
#   trigger            : semantic / trig_always          (keyword = main table)
#   retrieval window   : topk4 / topk16                  (k=8 = main table)
# Idempotent: a result file with an 'after' section is skipped. Restores xlam-2-8b.
#
# Usage: bash experiments/run_ablation_musique_v2.sh [model ...]  (default: all four models)
set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/ablation_musique_status.txt
FROZEN_MUS=$ROOT/research/frozen/musique_targets_59.json
KB_DIR=${MUSIQUE_KB_DIR:-$ROOT/data/chroma_musique}
KB_COLL=musique_kb
mkdir -p "$LOGDIR"
: > "$STATUS"
echo "KB_DIR=$KB_DIR" >> "$STATUS"

declare -A SPECS=(
  [xlam-2-8b]="/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r|xlam-2-8b|xlam|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja"
  [qwen3-8b]="/mnt/disk/cwh/LLMs/Qwen3-8B|qwen3-8b|hermes|--max-model-len 16384 --reasoning-parser qwen3"
  [gpt-oss-20b]="/mnt/disk/cwh/LLMs/gpt-oss-20b|gpt-oss-20b|openai|--max-num-seqs 64 --max-model-len 32768"
  [llama-3.1-8b]="/mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|llama3_json|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja"
)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("xlam-2-8b" "qwen3-8b" "gpt-oss-20b" "llama-3.1-8b")

serve () {  # $1 path  $2 served_name  $3 tool_parser  $4 extra flags
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  # shellcheck disable=SC2086
  nohup "$PY" "$VLLM" serve "$1" --port 8000 \
      --gpu-memory-utilization 0.85 \
      --served-model-name "$2" --enable-auto-tool-choice --tool-call-parser "$3" $4 \
      > "$LOGDIR/vllm_$2.log" 2>&1 &
  for _ in $(seq 1 150); do
    if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "\"id\": *\"$2\""; then
      echo "server up: $2"
      return 0
    fi
    sleep 5
  done
  echo "SERVER FAILED: $2 (see $LOGDIR/vllm_$2.log)"
  return 1
}

tool_smoke () {  # $1 served_name
  "$PY" - "$1" <<'PYEOF'
import sys, openai
name = sys.argv[1]
c = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
r = c.chat.completions.create(
    model=name,
    messages=[{"role": "user", "content": "What is the current weather in Paris? You must call the get_weather tool."}],
    tools=tools, temperature=0, max_tokens=200,
)
msg = r.choices[0].message
if not msg.tool_calls:
    print("TOOL_CALL_MISSING:", repr(msg.content)[:200]); sys.exit(2)
print("TOOL_CALL_OK")
PYEOF
}

run_arm () {  # $1 model  $2 arm  $3... 08 args
  local model=$1 arm=$2; shift 2
  local mname="08_longtail_musique"
  [ "$model" != "xlam-2-8b" ] && mname="08_longtail_${model}_musique"
  local out="$ROOT/results/08_longtail_${model}_${arm}_musique.json"
  if [ -s "$out" ] && "$PY" -c "import json,sys
try:
    d=json.load(open('$out')); v=d.get('variants',{})
    sys.exit(0 if any(isinstance(x, dict) and 'after' in x and 'co_retrieval' in x for x in v.values()) else 1)
except Exception:
    sys.exit(1)" 2>/dev/null; then
    echo "SKIP (done): $out"
    return
  fi
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="${model}_${arm}_mus_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  export AGENTIC_RAG_RUN_ID="$run_id"
  echo "=== $(date '+%F %T') arm $model/$arm (musique, clean-from $mname) ===" | tee -a "$STATUS"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py "$@" \
      --model "$model" \
      --target-records "$FROZEN_MUS" \
      --kb-dir "$KB_DIR" --kb-collection "$KB_COLL" \
      --clean-from "$ROOT/results/$mname.json" \
      --results-name "08_longtail_${model}_${arm}_musique" \
      > "$rundir/08_${model}_${arm}_musique.log" 2>&1 ) \
    && echo "$model/$arm musique OK ($run_id)" >> "$STATUS" || echo "$model/$arm musique RUN_FAILED ($run_id)" >> "$STATUS"
  unset AGENTIC_RAG_RUN_ID
}

for model in "${MODELS[@]}"; do
  spec="${SPECS[$model]}"
  IFS='|' read -r path name parser extra <<< "$spec"
  if serve "$path" "$name" "$parser" "$extra"; then
    tool_smoke "$name" >> "$LOGDIR/toolsmoke_$name.log" 2>&1 \
      && "$PY" "$ROOT/experiments/harness_smoke.py" "$name" >> "$LOGDIR/harnesssmoke_$name.log" 2>&1 \
      || { echo "$name TOOL_CALL_FAILED" >> "$STATUS"; continue; }
    case "$name" in
      qwen3-8b) export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}' ;;
      *)        unset AGENTIC_RAG_CHAT_KWARGS ;;
    esac
    # poison dose
    run_arm "$name" vol2  --volume 2 --variants cluster
    run_arm "$name" vol4  --volume 4 --variants cluster
    run_arm "$name" vol6  --volume 6 --variants cluster
    # style diversity (all dose 8)
    run_arm "$name" embed_hybrid --volume 8 --variants embed_hybrid
    run_arm "$name" mono         --volume 8 --variants cluster_mono
    run_arm "$name" nodiv        --volume 8 --variants cluster_nodiv
    run_arm "$name" greedy       --volume 8 --variants cluster_greedy
    # trigger
    run_arm "$name" semantic     --volume 8 --variants cluster --trigger-kind semantic
    run_arm "$name" trig_always  --volume 8 --variants cluster --trigger-kind always --always-probability 1.0
    # victim retrieval window
    run_arm "$name" topk4        --volume 8 --variants cluster --top-k 4
    run_arm "$name" topk16       --volume 8 --variants cluster --top-k 16
  else
    echo "$name SERVER_FAILED" >> "$STATUS"
  fi
done

# restore the default xlam-2-8b server
pkill -f "vllm serve" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 8
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 5
done
nohup "$PY" "$VLLM" serve /mnt/disk/cwh/LLMs/xlam-2-8b-fc-r --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 16384 \
    --served-model-name xlam-2-8b --enable-auto-tool-choice --tool-call-parser xlam --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja \
    > "$LOGDIR/vllm_xlam-2-8b_restore.log" 2>&1 &
echo "restoring xlam-2-8b server"
echo "=== done $(date '+%F %T') ===" >> "$STATUS"
cat "$STATUS"