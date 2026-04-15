# Reproducing Gemma 4 Deployment on L40s GPUs

Step-by-step guide to deploy Gemma 4 31B-it and 26B-A4B-it using vLLM on NVIDIA L40s GPUs.

## 1. Hardware Requirements

- **GPUs:** 4x NVIDIA L40s (48GB VRAM each)
  - GPUs 0,1 -> Gemma 4 31B-it (~42GB VRAM per GPU)
  - GPUs 2,3 -> Gemma 4 26B-A4B-it (~42GB VRAM per GPU)
- **RAM:** 64GB+ recommended
- **Disk:** ~150GB free (model weights + venv)
- **Tested on:** Cyfuture Cloud GPU instance, Ubuntu 24.04

## 2. Prerequisites

```bash
# Verify NVIDIA drivers and CUDA
nvidia-smi          # Should show 4x L40s, driver 550+
nvcc --version      # CUDA 12.8+

# Required packages
sudo apt update && sudo apt install -y python3.10 python3.10-venv tmux git
```

## 3. Clone This Repo

```bash
git clone git@github.com:ansumandas441/gemma4.git
cd gemma4
```

## 4. Python Virtual Environment

```bash
python3.10 -m venv gemma4-venv
source gemma4-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The key packages installed are:
- `vllm==0.19.0` — inference server
- `torch==2.10.0` — PyTorch with CUDA 12.8
- `transformers==5.5.4` — model loading
- `huggingface_hub==1.10.2` — model downloads

## 5. HuggingFace Authentication

You need HuggingFace access to download Gemma models.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the Gemma license:
   - [Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it)
   - [Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Login:
```bash
huggingface-cli login
# Paste your token when prompted
```

## 6. Download Model Weights

```bash
# Gemma 4 31B-it (~60GB)
./download_gemma4.sh

# Gemma 4 26B-A4B-it (~50GB)
./download_26b.sh
```

Downloads go to `~/.cache/huggingface/hub/`. Ensure you have enough disk space.

## 7. Configure

### API Keys

```bash
cp .env.example .env
# Edit .env and set your desired API keys:
#   GEMMA4_31B_API_KEY="your-key-here"
#   GEMMA4_26B_API_KEY="your-key-here"
source .env
```

### Paths

If your home directory is not `/home/primetrace`, update these files:
- `start_gemma4.sh` — line 2 (venv path)
- `start_gemma4_26b.sh` — line 2 (venv path)
- `manage_models.sh` — the `get_config` function (script, log paths)

### GPU Assignment

Default layout for 4x L40s:

| GPUs | Model | Port |
|---|---|---|
| 0, 1 | Gemma 4 31B-it | 8000 |
| 2, 3 | Gemma 4 26B-A4B-it | 8010 |

To change GPU assignment, edit `CUDA_VISIBLE_DEVICES` in the respective start script.

If you only have 2 GPUs, deploy one model at a time.

## 8. Start Serving

```bash
# Source API keys
source .env

# Start both models
./manage_models.sh all start

# Or start individually
./manage_models.sh 31b start
./manage_models.sh 26b start
```

Models take ~90 seconds to load.

## 9. Verify

```bash
# Check status
./manage_models.sh all status

# Expected output: both RUNNING, API Responding

# GPU memory check
nvidia-smi
# Each GPU should show ~42GB / 48GB used

# Test 31B
curl -H "Authorization: Bearer $GEMMA4_31B_API_KEY" http://localhost:8000/v1/models

# Test 26B
curl -H "Authorization: Bearer $GEMMA4_26B_API_KEY" http://localhost:8010/v1/models

# Full inference test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GEMMA4_31B_API_KEY" \
  -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"Say hello"}],"max_tokens":64}'
```

## 10. vLLM Flags Reference

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size 2` | 2 GPUs per model | Splits model across 2x L40s (48GB each = 96GB total, enough for 31B in fp16) |
| `--max-model-len 16384` | 16K tokens | Balances context length with VRAM — higher values risk OOM |
| `--gpu-memory-utilization 0.90` | 90% | Reserves 10% VRAM headroom to prevent OOM during peak KV cache usage |
| `--kv-cache-dtype fp8` | FP8 | Halves KV cache memory vs fp16, allowing more concurrent requests |
| `--enforce-eager` | — | Disables CUDA graphs — required for multi-GPU stability on L40s |
| `--disable-custom-all-reduce` | — | Uses NCCL all-reduce instead of vLLM's custom kernel — more reliable on L40s |
| `--enable-auto-tool-choice` | 31B only | Enables function/tool calling support |
| `--reasoning-parser gemma4` | 31B only | Parses Gemma 4's reasoning/thinking output format |
| `--tool-call-parser gemma4` | 31B only | Parses Gemma 4's tool call output format |

### Environment Variables

| Variable | Value | Why |
|---|---|---|
| `NCCL_P2P_DISABLE=1` | Disable P2P | L40s don't support NVLink P2P — NCCL falls back to PCIe, avoiding hangs |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | spawn | Required for multi-GPU — fork can cause CUDA context issues |

## 11. Troubleshooting

### Model won't start / OOM
- Check `nvidia-smi` — another process may be using GPU memory
- Kill stale processes: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill -9 {}`
- Reduce `--max-model-len` (e.g., 8192) if still OOM
- Reduce `--gpu-memory-utilization` to 0.85

### NCCL errors / hangs on startup
- Verify `NCCL_P2P_DISABLE=1` is set
- Verify `--disable-custom-all-reduce` is in the start script
- Check GPU topology: `nvidia-smi topo -m`

### Slow startup (~90s is normal)
- First load downloads/compiles kernels — subsequent starts are faster
- vLLM with `--enforce-eager` skips CUDA graph compilation (which would be even slower)

### API returns 401
- Verify API key: `echo $GEMMA4_31B_API_KEY`
- Make sure you ran `source .env` before starting the server

### Only 2 GPUs available
- Deploy one model at a time
- Edit the start script: `CUDA_VISIBLE_DEVICES=0,1`
- Use `./manage_models.sh 31b start` (not `all`)
