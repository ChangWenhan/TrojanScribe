#!/bin/bash
# Serve gpt-oss-20b for poison-text generation (baseline PoisonedRAG).
exec /home/cwh/anaconda3/envs/agents/bin/vllm serve /mnt/disk/cwh/LLMs/gpt-oss-20b \
  --port 8000 --gpu-memory-utilization 0.85 --max-num-seqs 64 --max-model-len 32768 \
  --served-model-name gpt-oss-20b --enable-auto-tool-choice --tool-call-parser openai \
  > /mnt/disk/cwh/AgenticRAG/results/logs/vllm_gptoss_poison.log 2>&1