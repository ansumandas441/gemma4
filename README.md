# Gemma 4 - Multi-Model vLLM Server

Deploy Google's Gemma 4 models on 4x NVIDIA L40s GPUs using vLLM.

> For a full step-by-step setup guide on fresh hardware, see [REPRODUCE.md](REPRODUCE.md).

## Server Overview

| | Gemma 4 31B-it | Gemma 4 26B-A4B-it (MoE) | Gemma 4 E4B (Base) |
|---|---|---|---|
| **Model** | `google/gemma-4-31B-it` | `google/gemma-4-26B-A4B-it` | `google/gemma-4-E4B` |
| **Port** | 8000 | 8010 | 8020 |
| **GPUs** | 0, 1 | 2, 3 | 2, 3 |
| **tmux** | `gemma4` | `gemma4-26b` | `gemma4-e4b` |
| **Script** | `start_gemma4.sh` | `start_gemma4_26b.sh` | `start_gemma4_e4b.sh` |
| **API Key Env** | `GEMMA4_31B_API_KEY` | `GEMMA4_26B_API_KEY` | `GEMMA4_E4B_API_KEY` |
| **Context** | 16K | 16K | 32K |
| **Type** | Instruction-tuned | Instruction-tuned | Base (completion) |

> **Note:** 26B and E4B share GPUs 2,3 -- run one at a time, not both simultaneously.

## Repository Contents

| File | Description |
|---|---|
| `start_gemma4.sh` | vLLM serve script for 31B (GPUs 0,1, port 8000, tool calling enabled) |
| `start_gemma4_26b.sh` | vLLM serve script for 26B MoE (GPUs 2,3, port 8010) |
| `start_gemma4_e4b.sh` | vLLM serve script for E4B base (GPUs 2,3, port 8020) |
| `manage_models.sh` | Unified management: start/stop/restart/status/set-key/logs (31B/26B) |
| `download_gemma4.sh` | Download 31B weights from HuggingFace |
| `download_26b.sh` | Download 26B weights from HuggingFace |
| `download_e4b.sh` | Download E4B weights from HuggingFace |
| `requirements.txt` | Python dependencies (vllm 0.19.0, torch 2.10.0, etc.) |
| `.env.example` | Template for API key environment variables |
| `REPRODUCE.md` | Full reproduction guide for L40s GPUs |

## Quick Start

```bash
# 1. Set up API keys
cp .env.example .env
# Edit .env with your desired keys
source .env

# 2. Start 31B + E4B (or swap E4B for 26B)
./manage_models.sh 31b start
tmux new-session -d -s gemma4-e4b 'source .env && /home/primetrace/start_gemma4_e4b.sh 2>&1 | tee /home/primetrace/gemma4-e4b-serve.log'

# 3. Check status (~90s to load)
./manage_models.sh 31b status
curl -H "Authorization: Bearer $GEMMA4_E4B_API_KEY" http://localhost:8020/v1/models
```

## API Usage

### 31B (port 8000) -- Chat completions
```bash
curl http://<SERVER_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{
    "model": "google/gemma-4-31B-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### 26B MoE (port 8010) -- Chat completions
```bash
curl http://<SERVER_IP>:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{
    "model": "google/gemma-4-26B-A4B-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### E4B Base (port 8020) -- Text completions
```bash
curl http://<SERVER_IP>:8020/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -d '{
    "model": "google/gemma-4-E4B",
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

> E4B is a **base model** (not instruction-tuned) -- use `/v1/completions` for text completion, not `/v1/chat/completions`.

## Managing API Keys

```bash
# Set a new key (changes script and restarts server)
./manage_models.sh 31b set-key YOUR_NEW_KEY
./manage_models.sh 26b set-key YOUR_NEW_KEY

# Remove key (open access)
./manage_models.sh 31b remove-key
./manage_models.sh 26b remove-key
```

## Starting / Stopping Servers

```bash
# 31B and 26B via manage_models.sh
./manage_models.sh 31b start
./manage_models.sh 26b start
./manage_models.sh all start      # 31B + 26B
./manage_models.sh 31b stop
./manage_models.sh all status

# E4B (manual -- shares GPUs 2,3 with 26B, stop 26B first)
./manage_models.sh 26b stop
export GEMMA4_E4B_API_KEY="your-key"
tmux new-session -d -s gemma4-e4b '/home/primetrace/start_gemma4_e4b.sh 2>&1 | tee /home/primetrace/gemma4-e4b-serve.log'

# Stop E4B
tmux kill-session -t gemma4-e4b

# Tail E4B logs
tail -f /home/primetrace/gemma4-e4b-serve.log
```

## Management Script

```
./manage_models.sh {31b|26b|all} {start|stop|restart|status|set-key <KEY>|remove-key|logs}
```

## Health Checks

```bash
# GPU usage
nvidia-smi

# List tmux sessions
tmux ls

# Test APIs
curl -H "Authorization: Bearer <YOUR_KEY>" http://localhost:8000/v1/models   # 31B
curl -H "Authorization: Bearer <YOUR_KEY>" http://localhost:8010/v1/models   # 26B
curl -H "Authorization: Bearer <YOUR_KEY>" http://localhost:8020/v1/models   # E4B
```

## Notes

- Models load in ~90 seconds after start (E4B is faster, ~30s)
- 31B runs on GPUs 0,1 independently
- 26B and E4B share GPUs 2,3 -- only run one at a time
- The 26B-A4B-it is a Mixture-of-Experts model (26B total, 4B active per token)
- E4B is a base model with native audio support, effective 4B params with Per-Layer Embeddings
- E4B uses `/v1/completions` endpoint (not chat) since it's not instruction-tuned
- KV cache uses fp8 for memory efficiency
- Context: 16K for 31B/26B, 32K for E4B
- 31B has tool calling / function calling enabled (`--enable-auto-tool-choice`)
