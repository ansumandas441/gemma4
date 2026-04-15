#!/bin/bash
source /home/primetrace/gemma4-venv/bin/activate
export HF_TOKEN=$(cat /home/primetrace/.cache/huggingface/token)
hf download google/gemma-4-26B-A4B --exclude '*.gguf'
