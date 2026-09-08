#!/bin/bash
# Cross-model ablation F driver v2 (2026-09-08): inject model A -> victim
# model B on the shared 60 hotpot targets (cluster v8, keyword trigger).
# Diagonal (A==B) pairs are the main-table same-model rows — not run here.
#
# v2 replay: the poison texts are NOT regenerated. Attacker A's main-table run
# already persisted them (08_longtail.json / 08_longtail_<att>.json, variant
# cluster), so this driver replays those texts into the KB with --phase
# inject-from (no LLM calls, no attacker server needed) and then evaluates
# victim B with --phase eval-after (clean baseline read from B's main-table
# file). One serve session per victim.
# Idempotent: a pair whose eval-after result file exists is skipped. Ends by
# restoring the default victim server.
#
# Usage: bash experiments/run_cross_model.sh   (runs all 12 off-diagonal pairs)

set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/cross_model_status.txt
mkdir -p "$LOGDIR"
: > "$STATUS"

declare -A SPECS=(
  [xlam-2-8b]="/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r|xlam-2-8b|xlam|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja"
  [qwen3-8b]="/mnt/disk/cwh/LLMs/Qwen3-8B|qwen3-8b|hermes|--max-model-len 16384 --reasoning-parser qwen3"
  [gpt-oss-20b]="/mnt/disk/cwh/LLMs/gpt-oss-20b|gpt-oss-20b|openai|--max-num-seqs 64 --max-model-len 32768"
  [llama-3.1-8b]="/mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|llama3_json|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja"
)
MODELS=("xlam-2-8b" "qwen3-8b" "gpt-oss-20b" "llama-3.1-8b")

serve () {  # $1 path  $2 served_name  $3 tool_parser  $4 extra flags (string)
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

run_pair () {  # $1 att  $2 vic
  local att=$1 vic=$2
  local out="$ROOT/results/08_longtail_xsm_${att}_to_${vic}.json"
  if [ -s "$out" ]; then
    echo "SKIP (exists): $out"
    return
  fi
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="xsm_${att}_to_${vic}_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  echo "=== $(date '+%F %T') cross-model $att -> $vic ===" | tee -a "$STATUS"

  local vspec="${SPECS[$vic]}"; IFS='|' read -r vpath vname vparser vextra <<< "$vspec"
  # poison source = attacker's main-table run (headline file for xlam-2-8b,
  # 08_longtail_<att>.json otherwise)
  local src="08_longtail"
  [ "$att" != "xlam-2-8b" ] && src="08_longtail_${att}"
  local srcpath="$ROOT/results/${src}.json"
  if [ ! -s "$srcpath" ]; then
    echo "attacker main-table missing: $srcpath (run the main table first)" | tee -a "$STATUS"
    return
  fi

  # one serve session for the victim: replay poison, then evaluate
  if ! serve "$vpath" "$vname" "$vparser" "$vextra"; then
    echo "$vic SERVER_FAILED" >> "$STATUS"; return
  fi
  tool_smoke "$vname" >> "$LOGDIR/toolsmoke_$vname.log" 2>&1 \
      && "$PY" "$ROOT/experiments/harness_smoke.py" "$vname" >> "$LOGDIR/harnesssmoke_$vname.log" 2>&1 \
      || { echo "$vic TOOL_CALL_FAILED" >> "$STATUS"; return; }
  case "$vname" in
    qwen3-8b) export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}' ;;
    *)        unset AGENTIC_RAG_CHAT_KWARGS ;;
  esac
  export AGENTIC_RAG_RUN_ID="$run_id"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
      --phase inject-from --inject-from "$srcpath" \
      --results-name "08_longtail_xsm_${att}_to_${vic}_inj" \
      > "$rundir/inject_${att}_to_${vic}.log" 2>&1 ) \
    && echo "$att poison replay OK" >> "$STATUS" || echo "$att poison replay RUN_FAILED" >> "$STATUS"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
      --targets 60 --volume 8 --variants cluster --use-shared-targets \
      --phase eval-after --model "$vname" \
      --results-name "08_longtail_xsm_${att}_to_${vic}" \
      > "$rundir/eval_${att}_to_${vic}.log" 2>&1 ) \
    && echo "$vic eval-after OK ($run_id)" >> "$STATUS" || echo "$vic eval-after RUN_FAILED ($run_id)" >> "$STATUS"
  unset AGENTIC_RAG_RUN_ID
}

for att in "${MODELS[@]}"; do
  for vic in "${MODELS[@]}"; do
    [ "$att" = "$vic" ] && continue   # diagonal = main-table same-model rows
    run_pair "$att" "$vic"
  done
done

# restore the default GLM server
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