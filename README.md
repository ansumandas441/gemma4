# Gemma 4 - Dual Model vLLM Server

Deploy Google's Gemma 4 31B-it and Gemma 4 26B-A4B-it (MoE) on 4x NVIDIA L40s GPUs using vLLM.

> For a full step-by-step setup guide on fresh hardware, see [REPRODUCE.md](REPRODUCE.md).

## Server Overview

| | Gemma 4 31B-it | Gemma 4 26B-A4B-it (MoE) |
|---|---|---|
| **Model** | `google/gemma-4-31B-it` | `google/gemma-4-26B-A4B-it` |
| **Port** | 8000 | 8010 |
| **GPUs** | 0, 1 | 2, 3 |
| **tmux** | `gemma4` | `gemma4-26b` |
| **Script** | `start_gemma4.sh` | `start_gemma4_26b.sh` |
| **API Key Env** | `GEMMA4_31B_API_KEY` | `GEMMA4_26B_API_KEY` |

## Repository Contents

| File | Description |
|---|---|
| `start_gemma4.sh` | vLLM serve script for 31B (GPUs 0,1, port 8000, tool calling enabled) |
| `start_gemma4_26b.sh` | vLLM serve script for 26B MoE (GPUs 2,3, port 8010) |
| `manage_models.sh` | Unified management: start/stop/restart/status/set-key/logs |
| `download_gemma4.sh` | Download 31B weights from HuggingFace |
| `download_26b.sh` | Download 26B weights from HuggingFace |
| `requirements.txt` | Python dependencies (vllm 0.19.0, torch 2.10.0, etc.) |
| `.env.example` | Template for API key environment variables |
| `REPRODUCE.md` | Full reproduction guide for L40s GPUs |

## Quick Start

```bash
# 1. Set up API keys
cp .env.example .env
# Edit .env with your desired keys
source .env

# 2. Start both models
./manage_models.sh all start

# 3. Check status (~90s to load)
./manage_models.sh all status
```

## API Usage

### 31B (port 8000)
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

### 26B MoE (port 8010)
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
# Start
./manage_models.sh 31b start      # 31B only
./manage_models.sh 26b start      # 26B only
./manage_models.sh all start      # Both

# Stop
./manage_models.sh 31b stop
./manage_models.sh 26b stop
./manage_models.sh all stop

# Restart
./manage_models.sh 26b restart

# Status
./manage_models.sh all status

# Tail logs
./manage_models.sh 31b logs       # Ctrl+C to exit
```

## Management Script

```
./manage_models.sh {31b|26b|all} {start|stop|restart|status|set-key <KEY>|remove-key|logs}
```

### Manual Management (without script)

```bash
# Start
tmux new-session -d -s gemma4 '/home/primetrace/start_gemma4.sh 2>&1 | tee /home/primetrace/gemma4-serve.log'
tmux new-session -d -s gemma4-26b '/home/primetrace/start_gemma4_26b.sh 2>&1 | tee /home/primetrace/gemma4-26b-serve.log'

# Stop
tmux kill-session -t gemma4
tmux kill-session -t gemma4-26b
```

## Health Checks

```bash
# GPU usage (all 4 GPUs should show ~42GB used)
nvidia-smi

# List tmux sessions
tmux ls

# Test APIs
curl -H "Authorization: Bearer <YOUR_KEY>" http://localhost:8000/v1/models   # 31B
curl -H "Authorization: Bearer <YOUR_KEY>" http://localhost:8010/v1/models   # 26B
```

## Notes

- Both models load in ~90 seconds after start
- Models run independently on separate GPU pairs -- stopping one does not affect the other
- The 26B-A4B-it is a Mixture-of-Experts model (26B total, 4B active per token) -- faster inference
- KV cache uses fp8 for memory efficiency
- Context length: 16,384 tokens for both
- 31B has tool calling / function calling enabled (`--enable-auto-tool-choice`)
