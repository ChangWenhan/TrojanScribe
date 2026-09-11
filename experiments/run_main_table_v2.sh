#!/bin/bash
# Main-table driver v2 (2026-09-08): 4 backbones x {hotpot, musique}.
# Backbones: xlam-2-8b (headline), qwen3-8b, gpt-oss-20b, llama-3.1-8b.
# Per model: serve on :8000 -> tool-call smoke test -> hotpot run
# (--use-shared-targets, shared 60 targets + shared wrongs) -> musique run
# (--target-records frozen 59-target set). Idempotent: a run whose result
# file already exists is skipped. At the end the xlam-2-8b server is restored.
#
# Usage: bash experiments/run_main_table_v2.sh [spec ...]
#   spec = path|served_name|tool_parser|extra_vllm_flags (may be empty)

set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/main_table_status.txt
FROZEN_MUS=$ROOT/research/frozen/musique_targets_59.json
mkdir -p "$LOGDIR"
: > "$STATUS"

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

tool_smoke () {  # $1 served_name — structured tool_calls + clean content
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
plain = c.chat.completions.create(
    model=name,
    messages=[{"role": "user", "content": "Reply with exactly: PARIS"}],
    temperature=0, max_tokens=512,
)
content = plain.choices[0].message.content or ""
if " thinking" in content or " response" in content:
    print("THINK_LEAK:", repr(content)[:200]); sys.exit(3)
tight = c.chat.completions.create(
    model=name,
    messages=[{"role": "user", "content": "Reply with exactly: PARIS"}],
    temperature=0, max_tokens=64,
)
if not (tight.choices[0].message.content or "").strip():
    print("THINK_BURNS_BUDGET (empty content at max_tokens=64)"); sys.exit(4)
print("TOOL_CALL_OK content:", repr(content)[:80])
PYEOF
}

skip_if_done () {  # $1 result file -> 0 only when a variant has after answers
                   # (08 saves incrementally after clean, so an existing file
                   # does NOT mean the run finished)
  [ -s "$1" ] || return 1
  "$PY" -c "import json,sys
try:
    d=json.load(open('$1')); v=d.get('variants',{})
    sys.exit(0 if any('after' in x for x in v.values()) else 1)
except Exception:
    sys.exit(1)" && echo "SKIP (done): $1" && return 0
  return 1
}

run_model () {  # $1 path $2 name $3 parser $4 extra
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="$2_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  export AGENTIC_RAG_RUN_ID="$run_id"
  echo "=== $(date '+%F %T') model $2 (run $run_id) ===" | tee -a "$STATUS"
  case "$2" in
    qwen3-8b) export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}' ;;
    *)        unset AGENTIC_RAG_CHAT_KWARGS ;;
  esac
  # headline naming: xlam hotpot = 08_longtail.json, xlam musique = 08_longtail_musique.json
  local hname="08_longtail"
  [ "$2" != "xlam-2-8b" ] && hname="08_longtail_$2"
  local mname="08_longtail_musique"
  [ "$2" != "xlam-2-8b" ] && mname="08_longtail_${2}_musique"
  if serve "$1" "$2" "$3" "$4"; then
    if tool_smoke "$2" >> "$LOGDIR/toolsmoke_$2.log" 2>&1 \
        && "$PY" "$ROOT/experiments/harness_smoke.py" "$2" >> "$LOGDIR/harnesssmoke_$2.log" 2>&1; then
      if ! skip_if_done "$ROOT/results/$hname.json"; then
        ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
            --targets 60 --volume 8 --variants cluster \
            --use-shared-targets --model "$2" \
            --results-name "$hname" \
            > "$rundir/08_$2.log" 2>&1 ) \
          && echo "$2 hotpot OK" >> "$STATUS" || echo "$2 hotpot RUN_FAILED" >> "$STATUS"
      fi
      if ! skip_if_done "$ROOT/results/$mname.json"; then
        ( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
            --volume 8 --variants cluster \
            --target-records "$FROZEN_MUS" \
            --kb-dir "$ROOT/data/chroma_musique" --kb-collection musique_kb \
            --model "$2" \
            --results-name "$mname" \
            > "$rundir/08_${2}_musique.log" 2>&1 ) \
          && echo "$2 musique OK" >> "$STATUS" || echo "$2 musique RUN_FAILED" >> "$STATUS"
      fi
    else
      echo "$2 TOOL_CALL_FAILED" >> "$STATUS"
    fi
  else
    echo "$2 SERVER_FAILED" >> "$STATUS"
  fi
  unset AGENTIC_RAG_RUN_ID
}

MODELS_DEFAULT=(
  "/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r|xlam-2-8b|xlam|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja"
  "/mnt/disk/cwh/LLMs/Qwen3-8B|qwen3-8b|hermes|--max-model-len 16384 --reasoning-parser qwen3"
  "/mnt/disk/cwh/LLMs/gpt-oss-20b|gpt-oss-20b|openai|--max-num-seqs 64 --max-model-len 32768"
  # Llama-3.1 official template only renders single-tool history; the patched
  # chat_template_multitool.jinja is required for multi-write batches
  "/mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|llama3_json|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja"
)
SPECS=("$@")
if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=("${MODELS_DEFAULT[@]}")
fi
for spec in "${SPECS[@]}"; do
  IFS='|' read -r path name parser extra <<< "$spec"
  [ -z "${path:-}" ] && continue
  run_model "$path" "$name" "${parser:-hermes}" "${extra:-}"
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