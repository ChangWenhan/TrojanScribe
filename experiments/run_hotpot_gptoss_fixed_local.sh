#!/bin/bash
# HotpotQA gpt-oss redo (A4 fix: reasoning_effort=low + payload budget 768) —
# LOCAL share of the dual-machine split:
#   main row (volume 8) -> flag for the 141 driver -> 5 arms -> gpt-oss->xlam pair.
# Waits for the 141 llama watcher to finish first. Forced reruns (the existing
# gpt-oss hotpot files are the known-broken versions).
set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/hotpot_gptoss_fixed_status.txt
FLAG=$ROOT/results/hotpot_gptoss_main_done.flag
EFFORT='{"reasoning_effort": "low"}'
mkdir -p "$LOGDIR"
rm -f "$FLAG"

echo "=== $(date '+%F %T') local hotpot gpt-oss redo armed; waiting for the 141 llama run ===" >> "$STATUS"
while pgrep -f "[r]un_llama_musique_remote_after_qwen3.sh" >/dev/null; do sleep 20; done
sleep 20

serve () {
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  nohup "$PY" "$VLLM" serve /mnt/disk/cwh/LLMs/gpt-oss-20b --port 8000 \
      --gpu-memory-utilization 0.85 --served-model-name gpt-oss-20b \
      --enable-auto-tool-choice --tool-call-parser openai \
      --max-num-seqs 64 --max-model-len 32768 \
      > "$LOGDIR/vllm_gpt-oss-20b.log" 2>&1 &
  for _ in $(seq 1 150); do
    curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "\"id\": *\"gpt-oss-20b\"" && return 0
    sleep 5
  done
  echo "SERVER FAILED: gpt-oss-20b" >> "$STATUS"
  return 1
}

tool_smoke () {
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

serve || exit 1
tool_smoke gpt-oss-20b >> "$LOGDIR/toolsmoke_gpt-oss-20b_hot_fixed.log" 2>&1 \
  && "$PY" "$ROOT/experiments/harness_smoke.py" gpt-oss-20b >> "$LOGDIR/harnesssmoke_gpt-oss-20b_hot_fixed.log" 2>&1 \
  || { echo "gpt-oss-20b TOOL_CALL_FAILED" >> "$STATUS"; exit 1; }

unset AGENTIC_RAG_CHAT_KWARGS
echo "=== $(date '+%F %T') probe gate ===" >> "$STATUS"
BUDGET=""
for B in 768 1024 1536; do
  PL=$LOGDIR/gptoss_hot_probe_gate_$B.log
  ( cd "$ROOT/experiments" && "$PY" payload_probe.py --model gpt-oss-20b \
      --via-backend --top-level-kwargs "$EFFORT" --budgets "$B" --n 8 ) > "$PL" 2>&1
  cat "$PL" >> "$STATUS"
  if grep -q "PASS" "$PL"; then BUDGET=$B; break; fi
done
if [ -z "$BUDGET" ]; then
  echo "=== PROBE_GATE_FAILED; server left up for inspection ===" >> "$STATUS"; exit 1
fi
export AGENTIC_RAG_TOP_LEVEL_KWARGS="$EFFORT"
export AGENTIC_RAG_PAYLOAD_MAX_TOKENS="$BUDGET"
echo "=== probe gate PASS at low+$BUDGET; starting main row ===" >> "$STATUS"

stamp=$(date +%Y%m%d_%H%M%S)
run_id="gptoss_main_hot_fixed_${stamp}"
rundir="$ROOT/results/runs/$run_id"
mkdir -p "$rundir"
export AGENTIC_RAG_RUN_ID="$run_id"
echo "=== $(date '+%F %T') gpt-oss/main (hotpot FIXED low+$BUDGET) ===" >> "$STATUS"
( cd "$ROOT/experiments" && "$PY" longtail_attack.py \
    --targets 60 --volume 8 --variants cluster --use-shared-targets \
    --model gpt-oss-20b \
    --results-name "08_longtail_gpt-oss-20b" \
    > "$rundir/08_gpt-oss-20b_main.log" 2>&1 ) \
  && { echo "gpt-oss/main hotpot OK ($run_id)" >> "$STATUS"; touch "$FLAG"; } \
  || { echo "gpt-oss/main hotpot RUN_FAILED ($run_id)" >> "$STATUS"; exit 1; }
unset AGENTIC_RAG_RUN_ID

# local share of the 11 arms (isolated KB: data/chroma)
FORCE=1 bash "$ROOT/experiments/run_ablation_split.sh" \
  http://localhost:8000/v1 "$ROOT/data/chroma" "$STATUS" \
  vol2 vol4 vol6 embed_hybrid mono

# gpt-oss -> xlam cross-model pair (local victim; the split script serves xlam
# and restores it at the end)
FORCE=1 bash "$ROOT/experiments/run_cross_model_split.sh" local \
  http://localhost:8000/v1 "$ROOT/data/chroma" "$STATUS" gpt-oss-20b:xlam-2-8b

( cd "$ROOT/experiments" && "$PY" summarize_ablation.py ) >> "$STATUS" 2>&1 \
  && echo "=== summary refreshed ===" >> "$STATUS"
echo "=== $(date '+%F %T') local hotpot gpt-oss redo done ===" >> "$STATUS"
