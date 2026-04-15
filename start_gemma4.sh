#!/bin/bash
source /home/primetrace/gemma4-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --api-key "${GEMMA4_31B_API_KEY}" \
  --host 0.0.0.0 \
  --port 8000
