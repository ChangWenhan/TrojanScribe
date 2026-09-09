#!/bin/bash
# run_agentpoison_victims.sh — AgentPoison(adapted) baseline on the MAIN-TABLE victims.
#
# Poison texts are fixed (baseline/agentpoison/gen_agentpoison_writes.py: qwen3-8b
# trigger sampling + bge retrieval score). Evaluated against each main-table victim
# (xlam-2-8b / qwen3-8b / gpt-oss-20b / llama-3.1-8b), one per row, same protocol
# as the cluster main table: poison already injected (60 chunks), eval-after per
# victim using that victim's own clean baseline.
set -u
ROOT=/mnt/disk/cwh/AgenticRAG
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
LOG=$ROOT/results/logs
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
export AGENTIC_RAG_ASK_WORKERS=8
export AGENTIC_RAG_BASE_URL=http://localhost:8000/v1
TARGET_REC=$ROOT/results/baseline_agentpoison_targets.json
VICTIMS=(xlam-2-8b qwen3-8b gpt-oss-20b llama-3.1-8b)

declare -A SPECS=(
  [xlam-2-8b]="/mnt/disk/cwh/LLMs/xlam-2-8b-fc-r|xlam-2-8b|xlam|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja"
  [qwen3-8b]="/mnt/disk/cwh/LLMs/Qwen3-8B|qwen3-8b|hermes|--max-model-len 16384 --reasoning-parser qwen3"
  [gpt-oss-20b]="/mnt/disk/cwh/LLMs/gpt-oss-20b|gpt-oss-20b|openai|--max-num-seqs 64 --max-model-len 32768"
  [llama-3.1-8b]="/mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|llama3_json|--max-model-len 16384 --chat-template /mnt/disk/cwh/LLMs/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja"
)

serve_victim () {  # $1 victim name
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
  nohup "$PY" "$VLLM" serve "$vpath" --port 8000 \
      --gpu-memory-utilization 0.85 \
      --served-model-name "$vname" --enable-auto-tool-choice --tool-call-parser "$vparser" $vextra \
      > "$LOG/vllm_$vname.log" 2>&1 &
  for _ in $(seq 1 150); do
    curl -s --max-time 3 "http://localhost:8000/v1/models" 2>/dev/null | grep -q "\"id\": *\"$vname\"" && return 0
    sleep 5
  done
  echo "FAILED to serve $vname" >&2; return 1
}

for V in "${VICTIMS[@]}"; do
  OUT="$ROOT/results/baseline_agentpoison_eval_${V}.json"
  if [ -s "$OUT" ]; then
    echo "=== skip $V (already done) ==="
    continue
  fi
  echo "=== serve victim $V ==="
  serve_victim "$V" || { echo "serve failed for $V"; continue; }
  echo "=== eval-after: $V ==="
  if [ "$V" = "qwen3-8b" ]; then
    export AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}'
  else
    unset AGENTIC_RAG_CHAT_KWARGS
  fi
  ( cd "$ROOT/experiments" && AGENTIC_RAG_RUN_ID="baseline_ap_main_${V}" \
      "$PY" longtail_attack.py --phase eval-after --model "$V" \
      --target-records "$TARGET_REC" --results-name "baseline_agentpoison_eval_${V}" 2>&1 | tail -2 )
  echo "=== done $V ==="
done

echo "=== all victims done ==="