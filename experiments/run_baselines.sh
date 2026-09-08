#!/bin/bash
# ReAct baselines on the default victim (xlam-2-8b): serve -> smoke ->
# experiments/14_react_baselines.py (clean + naive + poisonedRAG + ours +
# topicattack -> hotpotqa_seed1_<method>_xlam28b.json).
set -u
export PYTHONUNBUFFERED=1
PY=/home/cwh/anaconda3/envs/agents/bin/python
VLLM=/home/cwh/anaconda3/envs/agents/bin/vllm
ROOT=/mnt/disk/cwh/AgenticRAG
LOGDIR=$ROOT/results/logs
STATUS=$ROOT/results/baseline_status.txt
mkdir -p "$LOGDIR"
: > "$STATUS"

serve () {
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
  done
  setsid "$PY" "$VLLM" serve /mnt/disk/cwh/LLMs/xlam-2-8b-fc-r --port 8000 \
      --gpu-memory-utilization 0.85 --max-model-len 16384 \
      --served-model-name xlam-2-8b --enable-auto-tool-choice --tool-call-parser xlam --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja \
      > "$LOGDIR/vllm_xlam-2-8b.log" 2>&1 < /dev/null &
  for _ in $(seq 1 150); do
    if curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q '"id": *"xlam-2-8b"'; then
      echo "server up"; return 0
    fi
    sleep 5
  done
  echo "SERVER_FAILED"; return 1
}

echo "=== $(date '+%F %T') react baselines (xlam-2-8b) ===" | tee -a "$STATUS"
if serve; then
  ( cd "$ROOT/experiments" && "$PY" 14_react_baselines.py > "$LOGDIR/14_react_baselines.log" 2>&1 ) \
    && echo "baselines OK" >> "$STATUS" || echo "baselines RUN_FAILED" >> "$STATUS"
else
  echo "SERVER_FAILED" >> "$STATUS"
fi
echo "=== done $(date '+%F %T') ===" >> "$STATUS"
cat "$STATUS"