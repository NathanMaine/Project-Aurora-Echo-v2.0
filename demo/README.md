# Aurora Echo Demo — Quick Setup

Self-contained demo setup for running Aurora Echo on a workstation with an NVIDIA GPU and an OpenRouter API key.

## Tested On

- **ThinkPad P50** — 96 GB RAM, NVIDIA Quadro (4 GB VRAM)
- **Whisper model**: `small` (~1 GB VRAM)
- **LLM**: OpenRouter (any model — Llama 3.1, GPT-4o, Claude, etc.)

## Setup (3 steps)

```bash
# 1. Run setup
cd demo
chmod +x setup.sh run.sh
./setup.sh

# 2. Set your OpenRouter API key
#    Edit ../.env and replace the placeholder with your key
#    Get a key at https://openrouter.ai/keys

# 3. Start the server
./run.sh
```

Then open **http://localhost:8000** in your browser.

## What You Get

1. Click "Record" in the browser UI
2. Speak into your microphone
3. Click "Stop"
4. Aurora Echo will:
   - Transcribe your speech (faster-whisper on GPU)
   - Summarize the meeting (via OpenRouter LLM)
   - Extract action items with assignees and due dates

## Changing the LLM Model

Edit `../.env` and change `OPENAI_MODEL_ID`:

```bash
# Free/cheap models on OpenRouter:
OPENAI_MODEL_ID=meta-llama/llama-3.1-8b-instruct
OPENAI_MODEL_ID=google/gemma-2-9b-it

# Premium models:
OPENAI_MODEL_ID=openai/gpt-4o-mini
OPENAI_MODEL_ID=anthropic/claude-3.5-sonnet
```

## Changing the Whisper Model

Edit `../.env` and change `WHISPER_MODEL_SIZE`:

| Model | VRAM | Accuracy | Speed |
|-------|------|----------|-------|
| `tiny` | ~0.5 GB | Low | Fastest |
| `base` | ~0.5 GB | Fair | Fast |
| `small` | ~1 GB | Good | Moderate |
| `medium` | ~2.5 GB | Better | Slower |
| `large-v2` | ~4 GB | Best | Slowest |

For a 4 GB GPU, use `small` (default) or `medium`.

## Optional: Speaker Diarization

To identify who said what, get a Hugging Face token and add to `../.env`:

```bash
HF_TOKEN=hf_your-token-here
```

Then uncomment `pyannote.audio` in `demo/requirements.txt` and re-run `./setup.sh`.

## Files

```
demo/
├── README.md          # This file
├── .env.example       # Template config for OpenRouter
├── requirements.txt   # Lightweight deps (no TensorRT)
├── setup.sh           # One-shot setup script
└── run.sh             # Start the server
```

All demo files are self-contained. The main repo files are unchanged.
