#!/bin/bash
# serve_141.sh — remotely serve a victim model on 192.168.31.141 (hp user).
# Usage: sshpass -p '...' ssh hp@192.168.31.141 'bash -s' < serve_141.sh <model>
set -u
MODEL="${1:-qwen3-8b}"
VLLM=/home/hp/anaconda3/envs/vllm/bin/vllm
LLMS=/home/hp/LLMs
case "$MODEL" in
  qwen3-8b)
    PATH_ARG="$LLMS/Qwen3-8B"; NAME=qwen3-8b; PARSER=hermes; EXTRA="--max-model-len 16384 --reasoning-parser qwen3"
    ;;
  llama-3.1-8b)
    PATH_ARG="$LLMS/Meta-Llama-3.1-8B-Instruct"; NAME=llama-3.1-8b; PARSER=llama3_json; EXTRA="--max-model-len 16384 --chat-template $LLMS/Meta-Llama-3.1-8B-Instruct/chat_template_multitool.jinja"
    ;;
  gpt-oss-20b)
    PATH_ARG="$LLMS/gpt-oss-20b"; NAME=gpt-oss-20b; PARSER=openai; EXTRA="--max-num-seqs 64 --max-model-len 32768"
    ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac

# kill any existing vllm
pkill -f "vllm serve" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 8
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 5
done

setsid nohup "$VLLM" serve "$PATH_ARG" --port 8000 \
    --gpu-memory-utilization 0.85 \
    --served-model-name "$NAME" --enable-auto-tool-choice --tool-call-parser "$PARSER" $EXTRA \
    > /home/hp/vllm_$NAME.log 2>&1 &

for _ in $(seq 1 120); do
  curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q "\"id\": *\"$NAME\"" && { echo "SERVED $NAME"; exit 0; }
  sleep 5
done
echo "FAILED to serve $NAME" >&2
tail -5 /home/hp/vllm_$NAME.log 2>/dev/null
exit 1