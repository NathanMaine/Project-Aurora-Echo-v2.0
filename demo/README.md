# Aurora Echo Demo — Quick Setup

Self-contained demo setup for running Aurora Echo on your laptop or workstation with a cloud LLM provider.

## Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | None (CPU-only works) | NVIDIA GPU with 2+ GB VRAM |
| **Disk** | 5 GB free | 10 GB free (for model downloads) |
| **OS** | Linux, macOS, or Windows | Linux (best CUDA support) |
| **Python** | 3.10+ | 3.11+ |

**GPU notes:**
- No GPU? ASR runs on CPU — slower but fully functional.
- 2-4 GB VRAM? Use Whisper `small` (~1 GB) or `base` (~0.5 GB).
- 8+ GB VRAM? Use Whisper `large-v2` for best accuracy, or enable TensorRT via the main repo.

## Setup (3 steps)

```bash
# 1. Run setup
cd demo
chmod +x setup.sh run.sh
./setup.sh

# 2. Set your LLM API key
#    Edit ../.env and replace the placeholder with your key
#    (see "LLM Providers" below for options)

# 3. Start the server
./run.sh
```

Then open **http://localhost:8000** in your browser.

## What You Get

1. Click "Record" in the browser UI
2. Speak into your microphone
3. Click "Stop"
4. Aurora Echo will:
   - Transcribe your speech (faster-whisper, GPU-accelerated if available)
   - Summarize the meeting via your chosen LLM provider
   - Extract action items with assignees and due dates

## LLM Providers

Aurora Echo supports multiple LLM providers. Set one up in `../.env`:

### OpenRouter (recommended for getting started)

Access dozens of models through one API key. Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

```bash
LLM_PROVIDER_ORDER=openai
OPENAI_API_KEY=sk-or-v1-your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL_ID=meta-llama/llama-3.1-8b-instruct
```

Popular OpenRouter models:

| Model | Cost | Quality |
|-------|------|---------|
| `meta-llama/llama-3.1-8b-instruct` | Free/cheap | Good |
| `google/gemma-2-9b-it` | Free/cheap | Good |
| `openai/gpt-4o-mini` | ~$0.15/1M tokens | Very good |
| `anthropic/claude-3.5-sonnet` | ~$3/1M tokens | Excellent |
| `openai/gpt-4o` | ~$2.50/1M tokens | Excellent |

### OpenAI (direct)

```bash
LLM_PROVIDER_ORDER=openai
OPENAI_API_KEY=sk-your-openai-key
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL_ID=gpt-4o-mini
```

### Anthropic Claude (direct)

```bash
LLM_PROVIDER_ORDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key
ANTHROPIC_MODEL_ID=claude-3-sonnet-20240229
```

### Google Gemini (direct)

```bash
LLM_PROVIDER_ORDER=gemini
GOOGLE_GEMINI_API_KEY=your-key
GOOGLE_GEMINI_MODEL_ID=gemini-1.5-pro-latest
```

### xAI Grok (direct)

```bash
LLM_PROVIDER_ORDER=grok
XAI_API_KEY=xai-your-key
```

### Multiple providers (failover)

Chain providers so if one fails, the next is tried automatically:

```bash
LLM_PROVIDER_ORDER=openai,claude,gemini
OPENAI_API_KEY=sk-or-v1-your-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL_ID=meta-llama/llama-3.1-8b-instruct
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_GEMINI_API_KEY=your-key
```

### Local LLMs (no API key needed)

If you have enough VRAM (8+ GB) or want to run on CPU:

**Ollama:**
```bash
# Install Ollama: https://ollama.ai
# Pull a model: ollama pull llama3.1:8b
LLM_PROVIDER_ORDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_ID=llama3.1:8b
```

**vLLM** (requires 16+ GB VRAM):
```bash
LLM_PROVIDER_ORDER=vllm
VLLM_BASE_URL=http://localhost:8001
VLLM_MODEL_ID=meta-llama-3-8b-instruct
```

## Changing the Whisper Model

Edit `../.env` and set `WHISPER_MODEL_SIZE`:

| Model | VRAM | RAM (CPU) | Accuracy | Speed |
|-------|------|-----------|----------|-------|
| `tiny` | ~0.5 GB | ~1 GB | Low | Fastest |
| `base` | ~0.5 GB | ~1 GB | Fair | Fast |
| `small` | ~1 GB | ~2 GB | Good | Moderate |
| `medium` | ~2.5 GB | ~5 GB | Better | Slower |
| `large-v2` | ~4 GB | ~10 GB | Best | Slowest |

Pick based on your hardware:
- **No GPU**: `tiny` or `base` (CPU mode)
- **2-4 GB VRAM**: `small` (default) or `medium`
- **8+ GB VRAM**: `large-v2` for best accuracy

## Optional: Speaker Diarization

To identify who said what, get a [Hugging Face token](https://huggingface.co/settings/tokens) and add to `../.env`:

```bash
HF_TOKEN=hf_your-token-here
```

Then uncomment `pyannote.audio` in `demo/requirements.txt` and re-run `./setup.sh`.

## Files

```
demo/
├── README.md          # This file
├── .env.example       # Template config (OpenRouter default)
├── requirements.txt   # Lightweight deps (no TensorRT)
├── setup.sh           # One-shot setup script
└── run.sh             # Start the server
```

All demo files are self-contained. The main repo files are unchanged.
