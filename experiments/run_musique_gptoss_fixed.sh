#!/bin/bash
# gpt-oss-20b MuSiQue redo with the A4 generation-budget fix (2026-09-10):
#   AGENTIC_RAG_TOP_LEVEL_KWARGS='{"reasoning_effort": "low"}'  (harmony path
#     only reads reasoning_effort as a top-level request field)
#   AGENTIC_RAG_PAYLOAD_MAX_TOKENS=768..1536 (escalated if the probe is partial)
#
# Waits for the running xlam driver (run_ablation_musique_v2.sh) to finish its
# last setting, takes over the GPU, hard-gates on payload_probe.py, then reruns
# the gpt-oss main row (volume 8 cluster) + all 11 ablation settings on MuSiQue.
# Restores the xlam-2-8b server at the end.
#
# Idempotency intentionally ABSENT: this is a forced rerun; existing
# gpt-oss-*_musique.json files are the known-broken versions.
set -u
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_SAMPLE_WORKERS=8
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/ablation_musique_gptoss_status.txt
OLD_STATUS=$ROOT/results/ablation_musique_status.txt
FROZEN_MUS=$ROOT/research/frozen/musique_targets_59.json
KB_DIR=$ROOT/data/chroma_musique
KB_COLL=musique_kb
EFFORT='{"reasoning_effort": "low"}'
mkdir -p "$LOGDIR"

echo "=== $(date '+%F %T') fixed gpt-oss driver armed; waiting for xlam arms ===" >> "$STATUS"
while true; do
  if grep -q "xlam-2-8b/topk16 musique" "$OLD_STATUS" 2>/dev/null; then
    echo "=== $(date '+%F %T') detected xlam topk16 done; taking over ===" >> "$STATUS"
    break
  fi
  if grep -q "gpt-oss-20b" "$OLD_STATUS" 2>/dev/null; then
    echo "=== $(date '+%F %T') old driver already reached gpt-oss; taking over ===" >> "$STATUS"
    break
  fi
  if ! pgrep -f "[r]un_ablation_musique_v2.sh" >/dev/null; then
    if grep -q "xlam-2-8b/topk16 musique OK" "$OLD_STATUS" 2>/dev/null; then
      echo "=== $(date '+%F %T') old driver exited after topk16; taking over ===" >> "$STATUS"
      break
    fi
    echo "=== $(date '+%F %T') ABORT: old driver disappeared before finishing xlam arms ===" >> "$STATUS"
    cat "$OLD_STATUS" >> "$STATUS"
    exit 1
  fi
  sleep 20
done

# --- take over the GPU -------------------------------------------------------
pkill -f "[r]un_ablation_musique_v2.sh" 2>/dev/null
pkill -f "[l]ongtail_attack.py" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 8
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 5
done

serve () {  # $1 model dir  $2 served_name
  nohup "$PY" "$VLLM" serve "$1" --port 8000 \
      --gpu-memory-utilization 0.85 --served-model-name "$2" \
      --enable-auto-tool-choice --tool-call-parser openai \
      --max-num-seqs 64 --max-model-len 32768 \
      > "$LOGDIR/vllm_$2.log" 2>&1 &
  for _ in $(seq 1 150); do
    if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "\"id\": *\"$2\""; then
      echo "server up: $2" | tee -a "$STATUS"
      return 0
    fi
    sleep 5
  done
  echo "SERVER FAILED: $2" | tee -a "$STATUS"
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

if ! serve /mnt/disk/cwh/LLMs/gpt-oss-20b gpt-oss-20b; then
  echo "gpt-oss-20b SERVER_FAILED" >> "$STATUS"; exit 1
fi
tool_smoke gpt-oss-20b >> "$LOGDIR/toolsmoke_gpt-oss-20b_fixed.log" 2>&1 \
  && "$PY" "$ROOT/experiments/harness_smoke.py" gpt-oss-20b >> "$LOGDIR/harnesssmoke_gpt-oss-20b_fixed.log" 2>&1 \
  || { echo "gpt-oss-20b TOOL_CALL_FAILED" >> "$STATUS"; exit 1; }

# --- probe: diagnostics grid + hard gate with budget escalation --------------
unset AGENTIC_RAG_CHAT_KWARGS
echo "=== $(date '+%F %T') probe diagnostics (raw grid) ===" >> "$STATUS"
( cd "$ROOT/experiments" && "$PY" payload_probe.py --model gpt-oss-20b --n 8 ) \
  > "$LOGDIR/gptoss_payload_probe_grid.log" 2>&1
cat "$LOGDIR/gptoss_payload_probe_grid.log" >> "$STATUS"

BUDGET=""
for B in 768 1024 1536; do
  PL=$LOGDIR/gptoss_payload_probe_gate_$B.log
  ( cd "$ROOT/experiments" && "$PY" payload_probe.py --model gpt-oss-20b \
      --via-backend --top-level-kwargs "$EFFORT" --budgets "$B" --n 8 ) > "$PL" 2>&1
  cat "$PL" >> "$STATUS"
  if grep -q "PASS" "$PL"; then
    BUDGET=$B
    echo "=== probe gate PASS at reasoning_effort=low budget=$B ===" >> "$STATUS"
    break
  fi
done
if [ -z "$BUDGET" ]; then
  echo "=== PROBE_GATE_FAILED: low did not fill the pool at 768/1024/1536; server left up for inspection ===" >> "$STATUS"
  exit 1
fi
export AGENTIC_RAG_TOP_LEVEL_KWARGS="$EFFORT"
export AGENTIC_RAG_PAYLOAD_MAX_TOKENS="$BUDGET"

run_arm () {  # $1 arm ("main" = volume 8 cluster main row)  $2... longtail args
  local arm=$1; shift
  local results_name
  if [ "$arm" = "main" ]; then
    results_name="08_longtail_gpt-oss-20b_musique"
  else
    results_name="08_longtail_gpt-oss-20b_${arm}_musique"
  fi
  local stamp run_id rundir
  stamp=$(date +%Y%m%d_%H%M%S)
  run_id="gptoss_${arm}_mus_fixed_${stamp}"
  rundir="$ROOT/results/runs/$run_id"
  mkdir -p "$rundir"
  local clean_arg=()
  [ "$arm" != "main" ] && clean_arg=(--clean-from "$ROOT/results/08_longtail_gpt-oss-20b_musique.json")
  export AGENTIC_RAG_RUN_ID="$run_id"
  echo "=== $(date '+%F %T') gpt-oss/$arm (musique FIXED low+$BUDGET) ===" >> "$STATUS"
  ( cd "$ROOT/experiments" && "$PY" longtail_attack.py "$@" \
      --model gpt-oss-20b \
      --target-records "$FROZEN_MUS" \
      --kb-dir "$KB_DIR" --kb-collection "$KB_COLL" \
      "${clean_arg[@]}" \
      --results-name "$results_name" \
      > "$rundir/08_gpt-oss-20b_${arm}_musique.log" 2>&1 ) \
    && echo "gpt-oss/$arm musique OK ($run_id)" >> "$STATUS" \
    || echo "gpt-oss/$arm musique RUN_FAILED ($run_id)" >> "$STATUS"
  unset AGENTIC_RAG_RUN_ID
}

# main row first: the ablation arms reuse its clean baseline via --clean-from
run_arm main  --volume 8 --variants cluster
run_arm vol2  --volume 2 --variants cluster
run_arm vol4  --volume 4 --variants cluster
run_arm vol6  --volume 6 --variants cluster
run_arm embed_hybrid --volume 8 --variants embed_hybrid
run_arm mono         --volume 8 --variants cluster_mono
run_arm nodiv        --volume 8 --variants cluster_nodiv
run_arm greedy       --volume 8 --variants cluster_greedy
run_arm semantic     --volume 8 --variants cluster --trigger-kind semantic
run_arm trig_always  --volume 8 --variants cluster --trigger-kind always --always-probability 1.0
run_arm topk4        --volume 8 --variants cluster --top-k 4
run_arm topk16       --volume 8 --variants cluster --top-k 16

( cd "$ROOT/experiments" && "$PY" summarize_ablation.py ) >> "$STATUS" 2>&1 \
  && echo "=== summary refreshed ===" >> "$STATUS"

# --- restore the default xlam-2-8b server -----------------------------------
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
echo "=== $(date '+%F %T') done; xlam-2-8b restoring ===" >> "$STATUS"
cat "$STATUS"
