# Project Aurora Echo 2.0 — NVIDIA Accelerated AI Meeting Copilot

AI meeting copilot that streams audio from the browser, transcribes it with
GPU-accelerated ASR (faster-whisper with TensorRT optimization), diarises speakers
with `pyannote.audio`, and produces a structured summary using a pluggable LLM
provider stack. Everything is orchestrated asynchronously so the UI receives
status updates and partial transcripts in real time.

## Version 2.0 Highlights

- **NVIDIA DCGM Monitoring** — Real-time GPU metrics (utilization, memory, temperature, power) via Prometheus/Grafana.
- **TensorRT Optimization** — 2-5x faster ASR inference with TensorRT-optimized Whisper models.
- **Triton Inference Server** — Production-grade model serving with vLLM backend for LLM.
- **NVIDIA NIM Microservices** — Cloud-optimized inference for both LLM and ASR.
- **Enhanced Observability** — Advanced Grafana dashboards correlating GPU and application metrics.
- **8 LLM Providers** — vLLM, Ollama, NVIDIA NIM, OpenAI, Azure OpenAI, Anthropic Claude, Google Gemini, xAI Grok with automatic failover.
- **Hardened Security** — Non-root Docker container, encrypted audio buffers with size limits, no hardcoded secrets, sanitized error responses.
- **DRY Provider Architecture** — Centralized retry logic in `LLMProvider` base class with exponential backoff.

## Architecture

```
Browser (mic) ──WebSocket──▶ FastAPI (app.py)
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼              ▼
           SecureAudioBuffer  InferenceOrchestrator
           (Fernet encryption) (async worker queue)
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼              ▼
              ASR Service   Diarization    LLM Service
              (cascade)    (pyannote)     (failover chain)
                  │                            │
          ┌───────┼───────┐          ┌─────────┼─────────┐
          ▼       ▼       ▼          ▼         ▼         ▼
        NIM   TensorRT  faster-    vLLM    Ollama     Cloud
        ASR   Whisper   whisper    NIM     Triton   (OpenAI,
                                                    Claude,
                                                    Gemini,
                                                    Grok,
                                                    Azure)
```

## Feature Highlights

- **Binary WebSocket transport** — Browser sends raw PCM frames; server replies with status events, partial transcripts, and final JSON payloads.
- **Async inference pipeline** — Audio chunks queue through an `InferenceOrchestrator` with configurable workers and batch size. Queue has a bounded size to prevent memory exhaustion under load.
- **ASR backend cascade** — NIM ASR → TensorRT Whisper → faster-whisper → error, with automatic fallback.
- **Multi-provider LLM failover** — Configure provider priority via `LLM_PROVIDER_ORDER` env var. Each provider retries with exponential backoff before falling through to the next.
- **Speaker diarisation** — Opt-in via `HF_TOKEN`; runs in a background thread to avoid blocking the event loop. Falls back gracefully if the Hugging Face pipeline is unavailable.
- **Encrypted audio buffer** — Optional Fernet encryption for in-memory PCM data with a configurable size cap (default 100 MB) to prevent DoS.
- **Prometheus observability** — Counters/histograms on `/metrics` for ASR latency, LLM latency, diarisation latency, job duration, job failures, and queue depth, integrated with NVIDIA DCGM GPU metrics.
- **Optional TTS feedback** — Summaries can be spoken locally via `pyttsx3`.

## NVIDIA Technology Stack

| Technology | Purpose | Documentation |
|------------|---------|---------------|
| **NVIDIA DCGM** | GPU Monitoring | [README-dcgm.md](README-dcgm.md) |
| **TensorRT** | ASR Inference Optimization | [README-tensorrt.md](README-tensorrt.md) |
| **Triton Inference Server** | LLM Model Serving | [README-triton.md](README-triton.md) |
| **NVIDIA NIM** | Inference Microservices | [README-nim.md](README-nim.md) |

## LLM Providers

| Provider | Type | Config |
|----------|------|--------|
| **vLLM** | Local GPU | `VLLM_BASE_URL`, `VLLM_MODEL` |
| **Ollama** | Local | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| **NVIDIA NIM** | Cloud/On-prem | `NIM_API_KEY`, `NIM_MODEL` |
| **OpenAI** | Cloud | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| **Azure OpenAI** | Cloud | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| **Anthropic Claude** | Cloud | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| **Google Gemini** | Cloud | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| **xAI Grok** | Cloud | `XAI_API_KEY` |

Provider priority is set via `LLM_PROVIDER_ORDER` (comma-separated list of provider names).

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# Run the server
uvicorn app:app --host 127.0.0.1 --port 8000
```

### Docker Deployment

```bash
# Configure environment
cp .env.example .env.docker
# Edit .env.docker — you MUST set GRAFANA_ADMIN_PASSWORD

# Start base stack with DCGM monitoring
docker-compose up -d

# Optional: Add Triton for LLM serving
docker-compose -f docker-compose.yml -f docker-compose.triton.yml up -d

# Optional: Add NVIDIA NIM microservices
docker-compose -f docker-compose.yml -f docker-compose.nim.yml up -d
```

### Access Points

| Service | URL |
|---------|-----|
| UI | `https://localhost` (via Traefik) |
| API Health | `GET /test` |
| Prometheus Metrics | `GET /metrics` |
| Grafana | `http://localhost:3000` |
| Prometheus UI | `http://localhost:9090` |

## Project Structure

```
├── app.py                    # FastAPI app, WebSocket handler, pipeline wiring
├── services/
│   ├── asr_service.py        # ASR factory (NIM → TensorRT → faster-whisper)
│   ├── llm_service.py        # LLM failover chain across providers
│   ├── orchestrator.py       # Async inference queue with bounded workers
│   ├── audio_buffer.py       # Encrypted PCM accumulator with size limits
│   ├── tensorrt_whisper.py   # TensorRT-optimized Whisper backend
│   └── providers/
│       ├── base.py           # LLMProvider + ASRProvider abstract bases
│       ├── models.py         # Pydantic response models
│       ├── vllm_provider.py
│       ├── ollama_provider.py
│       ├── nim_provider.py
│       ├── nim_asr_provider.py
│       ├── openai_provider.py
│       ├── azure_openai_provider.py
│       ├── anthropic_provider.py
│       ├── gemini_provider.py
│       └── xai_grok_provider.py
├── observability.py          # Prometheus metrics definitions
├── docker/
│   ├── api.Dockerfile        # CUDA container (non-root user)
│   └── config/               # Traefik, Prometheus, Grafana configs
├── docker-compose.yml        # Main stack (API, Traefik, Prometheus, Grafana, DCGM)
├── benchmark_tensorrt.py     # TensorRT vs faster-whisper benchmarks
├── convert_to_tensorrt.py    # Model conversion script
└── requirements.txt
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER_ORDER` | No | Comma-separated provider priority (default: vllm,ollama,grok,...) |
| `HF_TOKEN` | No | Hugging Face token for speaker diarisation |
| `AUDIO_ENCRYPTION_KEY` | No | Fernet key for in-memory audio encryption |
| `INFERENCE_WORKERS` | No | Number of async workers (default: 2) |
| `INFERENCE_BATCH_SIZE` | No | Batch size per worker (default: 1) |
| `MAX_AUDIO_BUFFER_BYTES` | No | Audio buffer size cap in bytes (default: 100 MB) |
| `ENABLE_TTS` | No | Enable text-to-speech (default: 1) |
| `TTS_RATE` | No | TTS speech rate (default: 180) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `GRAFANA_ADMIN_PASSWORD` | Docker | Required for Docker deployment |

## Performance

With NVIDIA GPU acceleration:

| Metric | Expected Range |
|--------|---------------|
| ASR Latency | 50-100ms (2-5x faster with TensorRT) |
| LLM Latency | 100-300ms (1.5-2x faster with Triton/NIM) |
| GPU Utilization | Up to 80% with DCGM monitoring |
| Throughput | 40-60 requests/second with batch processing |

## Version History

- **2.0.1** (Feb 2026) — Security hardening: bind address changed from `0.0.0.0` to `127.0.0.1` (localhost only). `/docs` and `/redoc` endpoints disabled. Use a reverse proxy for intentional LAN exposure.
- **2.0** — NVIDIA DCGM, TensorRT, Triton, NIM integrations. 8 LLM providers with centralized retry. Hardened security (non-root container, encrypted buffers, no leaked secrets). Enhanced observability.
- **1.0** — Initial proof-of-concept with faster-whisper, vLLM, basic observability.

## License

All rights reserved.
