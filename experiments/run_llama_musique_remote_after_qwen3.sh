#!/bin/bash
# Run the llama-3.1-8b MuSiQue ablation arms on 141, in parallel with the
# local gpt-oss redo.
#
# Sequencing (all three share the 141 GPU and the chroma_musique_141 clone):
#   1. wait for the qwen3 remote driver, the semantic rerun, and its python
#   2. switch 141 to llama-3.1-8b (SSH key auth, no SSHPASS needed)
#   3. run the 11 arms via run_ablation_musique_remote.sh
#   4. refresh the summary and restore qwen3-8b on 141
set -u
ROOT=/mnt/disk/cwh/AgenticRAG
PY=/home/cwh/anaconda3/envs/agents/bin/python
STATUS=$ROOT/results/ablation_musique_llama_status.txt
LOGDIR=$ROOT/results/logs
REMOTE_HOST=hp@192.168.31.141
REMOTE_URL=http://192.168.31.141:8000/v1
mkdir -p "$LOGDIR"

echo "=== $(date '+%F %T') llama remote run armed; waiting for qwen3 driver + semantic rerun ===" >> "$STATUS"
while pgrep -f "[r]un_ablation_musique_remote.sh" >/dev/null \
   || pgrep -f "[r]erun_qwen3_semantic_musique.sh" >/dev/null \
   || pgrep -f "[l]ongtail_attack.py.*qwen3-8b" >/dev/null; do
  sleep 20
done
sleep 30

echo "=== $(date '+%F %T') switching 141 to llama-3.1-8b ===" >> "$STATUS"
ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash -s llama-3.1-8b" \
    < "$ROOT/scripts/serve_141.sh" > "$LOGDIR/serve141_llama-3.1-8b.log" 2>&1
ok=0
for _ in $(seq 1 60); do
  if curl -s --max-time 3 "$REMOTE_URL/models" 2>/dev/null | grep -q "\"id\": *\"llama-3.1-8b\""; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != "1" ]; then
  echo "=== $(date '+%F %T') LLAMA SERVE FAILED — aborting (see serve141 log) ===" >> "$STATUS"
  exit 1
fi
echo "=== $(date '+%F %T') 141 serving llama-3.1-8b; running arms ===" >> "$STATUS"

bash "$ROOT/experiments/run_ablation_musique_remote.sh" llama-3.1-8b "$STATUS"

( cd "$ROOT/experiments" && "$PY" summarize_ablation.py ) >> "$STATUS" 2>&1 \
  && echo "=== summary refreshed ===" >> "$STATUS"
echo "=== $(date '+%F %T') llama arms done; restoring qwen3-8b on 141 ===" >> "$STATUS"
ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" "bash -s qwen3-8b" \
    < "$ROOT/scripts/serve_141.sh" > "$LOGDIR/serve141_qwen3-8b.log" 2>&1
echo "=== $(date '+%F %T') all done ===" >> "$STATUS"
