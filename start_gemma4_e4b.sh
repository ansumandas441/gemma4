#!/bin/bash
source /home/primetrace/gemma4-venv/bin/activate
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve google/gemma-4-E4B \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --api-key "${GEMMA4_E4B_API_KEY}" \
  --host 0.0.0.0 \
  --port 8020
