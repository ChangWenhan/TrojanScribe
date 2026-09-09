#!/bin/bash
# Serve xlam-2-8b (default victim) for baseline evaluation.
exec /home/cwh/anaconda3/envs/agents/bin/vllm serve /mnt/disk/cwh/LLMs/xlam-2-8b-fc-r \
  --port 8000 --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --served-model-name xlam-2-8b --enable-auto-tool-choice --tool-call-parser xlam \
  --chat-template /mnt/disk/cwh/LLMs/xlam_chat_template.jinja \
  > /mnt/disk/cwh/AgenticRAG/results/logs/vllm_xlam_baseline.log 2>&1