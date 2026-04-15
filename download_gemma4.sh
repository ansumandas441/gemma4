#!/bin/bash
source /home/primetrace/gemma4-venv/bin/activate
export HF_TOKEN=$(cat /home/primetrace/.cache/huggingface/token)
hf download google/gemma-4-31B-it --exclude '*.gguf'
