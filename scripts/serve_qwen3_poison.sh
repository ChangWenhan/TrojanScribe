#!/bin/bash
# Serve Qwen3-8B for poison-text generation (baseline PoisonedRAG).
# Kept as a file so the launching command line never matches its own pattern.
exec /home/cwh/anaconda3/envs/agents/bin/vllm serve /mnt/disk/cwh/LLMs/Qwen3-8B \
  --port 8000 --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --served-model-name qwen3-8b --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  > /mnt/disk/cwh/AgenticRAG/results/logs/vllm_qwen3_baseline.log 2>&1