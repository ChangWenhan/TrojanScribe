#!/bin/bash
# run_cross_model_split.sh — cross-model ablation F on an EXPLICIT pair subset
# against ONE vLLM endpoint with ONE dedicated KB dir. Two disjoint instances
# (e.g. local GPU + the 192.168.31.141 endpoint) can run in parallel; poison
# texts are replayed from the attacker's main-table file (--phase inject-from),
# no attacker server is needed.
#
# Usage:
#   local  mode: bash experiments/run_cross_model_split.sh local  <base_url> <kb_dir> <status_file> <pairs...>
#     — this instance OWNS the vllm server on base_url: kills/serves each
#       victim in turn and restores xlam-2-8b at the end.
#   remote mode: bash experiments/run_cross_model_split.sh remote <base_url> <kb_dir> <status_file> <pairs...>
#     — the server is managed externally (ssh); this instance only WAITS for
#       the expected victim model to appear on base_url (30 min timeout per
#       group) and never touches the server process.
#
# Pairs are "att:vic" off-diagonal combos, e.g. xlam-2-8b:qwen3-8b. Order
# pairs so that consecutive pairs share the victim (one serve per victim).
set -u
ROOT=/mnt/disk/cwh/AgenticRAG
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
LOGDIR=$ROOT/results/logs
mkdir -p "$LOGDIR"
MODE="${1:?usage: run_cross_model_split.sh <local|remote> <base_url> <kb_dir> <status_file> <att:vic>...}"; shift
BASE_URL="${1:?}"; shift
KB_DIR="${1:?}"; shift
STATUS="${1:?}"; shift
[ $# -gt 0 ] || { echo "no pairs given" >> "$STATUS"; exit 1; }
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
export AGENTIC_RAG_BASE_URL="$BASE_URL"

declare -A SPECS=(
  [xlam-2-8b]="/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r|xlam-2-8b|xlam|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja"
  [qwen3-8b]="/mnt/disk/cwh/LLMs/Qwen3-8B|qwen3-8b|hermes|--max-model-len 16384 --reasoning-parser qwen3"
  [gpt-oss-20b]="/mnt/disk/cwh/LLMs/gpt-oss-20b|gpt-oss-20b|openai|--max-num-seqs 64 --max-model-len 32768"
  [llama-3.1-8b]="/mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|llama3_json|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja"
)

serve_local () {  # $1 victim name
  local vpath vname vparser vextra
  IFS='|' read -r vpath vname vparser vextra <<< "${SPECS[$1]}"
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  # shellcheck disable=SC2086
  nohup "$PY" "$VLLM" serve "$vpath" --port 8000 \
      --gpu-memory-utilization 0.85 \
      --served-model-name "$vname" --enable-auto-tool-choice --tool-call-parser "$vparser" $vextra \
      > "$LOGDIR/vllm_$vname.log" 2>&1 &
  for _ in $(seq 1 150); do
    curl -s --max-time 3 "$BASE_URL/models" 2>/dev/null | grep -q "\"id\": *\"$vname\"" && return 0
    sleep 5
  done
  echo "SERVER FAILED: $vname (see $LOGDIR/vllm_$vname.log)" >> "$STATUS"
  return 1
}

wait_remote () {  # $1 victim name — poll until this model is up on base_url
  for _ in $(seq 1 360); do
    curl -s --max-time 5 "$BASE_URL/models" 2>/dev/null | grep -q "\"id\": *\"$1\"" && return 0
    sleep 5
  done
  echo "REMOTE SERVER TIMEOUT: $1 not on $BASE_URL after 30min" >> "$STATUS"
  return 1
}

tool_smoke () {  # $1 served_name — uses $AGENTIC_RAG_BASE_URL (not localhost:
  # the 141-instance smoke must hit ITS endpoint, not the local GPU's server)
  "$PY" - "$1" <<'PYEOF'
import os, sys, openai
name = sys.argv[1]
c = openai.OpenAI(base_url=os.environ["AGENTIC_RAG_BASE_URL"], api_key="EMPTY")
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

run_pair () {  # $1 att  $2 vic  $3 cur_served_vic (in/out via CUR var)
  local att=$1 vic=$2
  local out="$ROOT/results/08_longtail_xsm_${att}_to_${vic}.json"
  if [ -s "$out" ]; then
    echo "SKIP (exists): $out" >> "$STATUS"
    return
  fi
  # (re)serve the victim when it changes
  if [ "${CUR:-}" != "$vic" ]; then
    if [ "$MODE" = local ]; then serve_local "$vic" || return; else wait_remote "$vic" || return; fi
    tool_smoke "$vic" >> "$LOGDIR/toolsmoke_${vic}_${MODE}.log" 2>&1 \
        && "$PY" "$ROOT/experiments/harness_smoke.py" "$vic" >> "$LOGDIR/harnesssmoke_${vic}_${MODE}.log" 2>&1 \
        || { echo "$vic TOOL_CALL_FAILED ($MODE)" >> "$STATUS"; return; }
    case "$vic" in
      qwen3-8b) export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}' ;;
      *)        unset AGENTIC_RAG_CHAT_KWARGS ;;
    esac
    CUR="$vic"
  fi
  local src="08_longtail"
  [ "$att" != "xlam-2-8b" ] && src="08_longtail_${att}"
  local srcpath="$ROOT/results/${src}.json"
  if [ ! -s "$srcpath" ]; then
    echo "attacker main-table missing: $srcpath" >> "$STATUS"
    return
  fi
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="xsm_${att}_to_${vic}_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  echo "=== $(date '+%F %T') cross-model $att -> $vic ($MODE, kb $(basename "$KB_DIR")) ===" | tee -a "$STATUS"
  export AGENTIC_RAG_RUN_ID="$run_id"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
      --phase inject-from --inject-from "$srcpath" --kb-dir "$KB_DIR" \
      --results-name "08_longtail_xsm_${att}_to_${vic}_inj" \
      > "$rundir/inject_${att}_to_${vic}.log" 2>&1 ) \
    && echo "$att poison replay OK" >> "$STATUS" || echo "$att poison replay RUN_FAILED" >> "$STATUS"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
      --targets 60 --volume 8 --variants cluster --use-shared-targets \
      --phase eval-after --model "$vic" --kb-dir "$KB_DIR" \
      --results-name "08_longtail_xsm_${att}_to_${vic}" \
      > "$rundir/eval_${att}_to_${vic}.log" 2>&1 ) \
    && echo "$vic eval-after OK ($run_id)" >> "$STATUS" || echo "$vic eval-after RUN_FAILED ($run_id)" >> "$STATUS"
  unset AGENTIC_RAG_RUN_ID
}

CUR=""
for pair in "$@"; do
  att="${pair%%:*}"; vic="${pair##*:}"
  run_pair "$att" "$vic"
done

if [ "$MODE" = local ]; then
  # restore the default local victim server
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
  echo "restoring xlam-2-8b server (local)"
fi
echo "=== split instance ($MODE) done $(date '+%F %T') ===" >> "$STATUS"
