# Project Aurora Echo 2.0 - NVIDIA Accelerated AI Meeting Copilot

AI meeting copilot that streams audio from the browser, transcribes it with
GPU-accelerated ASR (faster-whisper with TensorRT optimization), diarises speakers
with `pyannote.audio`, and produces a structured summary using a pluggable LLM
provider stack (local vLLM, NVIDIA NIM, Grok, OpenAI, etc.). Everything is
orchestrated asynchronously so the UI receives status updates and partial
transcripts in real time.

## 🚀 Version 2.0 Highlights: NVIDIA Technology Integration

- **NVIDIA DCGM Monitoring**: Real-time GPU metrics (utilization, memory, temperature, power) via Prometheus/Grafana.
- **TensorRT Optimization**: 2-5x faster ASR inference with TensorRT-optimized Whisper models.
- **Triton Inference Server**: Production-grade model serving with vLLM backend for LLM.
- **NVIDIA NIM Microservices**: Cloud-optimized inference microservices for LLM and ASR.
- **Enhanced Observability**: Advanced Grafana dashboards correlating GPU and application metrics.

## Feature Highlights
- **Binary WebSocket transport** – browser sends raw PCM frames, server replies
  with status events, partial transcripts, and final JSON payloads.
- **Async inference pipeline** – audio chunks queue through an
  `InferenceOrchestrator`, `ASRService` streams transcription with TensorRT acceleration,
  and the `SecureAudioBuffer` protects audio in memory (optional Fernet encryption).
- **Multi-provider LLM failover** – configure provider order via environment
  variables; supports NVIDIA NIM, local vLLM, Triton, Grok, OpenAI, etc.
- **Speaker diarisation** – opt-in via `HF_TOKEN`; falls back gracefully if the
  Hugging Face pipeline is unavailable.
- **Comprehensive Observability** – Prometheus counters/histograms exposed on `/metrics` for
  ASR, diarisation, and LLM latency plus queue depth and job totals, integrated with NVIDIA DCGM GPU metrics.
- **Optional TTS feedback** – summaries can be spoken locally via `pyttsx3`.

## NVIDIA Technology Stack

| Technology | Purpose | Status | Documentation |
|------------|---------|--------|---------------|
| **NVIDIA DCGM** | GPU Monitoring | ✅ Integrated | [README-dcgm.md](README-dcgm.md) |
| **TensorRT** | ASR Inference Optimization | ✅ Integrated | [README-tensorrt.md](README-tensorrt.md) |
| **Triton Inference Server** | LLM Model Serving | ✅ Integrated | [README-triton.md](README-triton.md) |
| **NVIDIA NIM** | Inference Microservices | ✅ Integrated | [README-nim.md](README-nim.md) |
| **Enhanced Observability** | Advanced Monitoring | ✅ Integrated | Built-in Grafana dashboards |

## Quick Start (Local with Docker)

1. **Clone the repository** (if not already done).
2. **Configure environment variables** – copy `.env.docker` and set necessary keys.
3. **Start the stack with NVIDIA integrations**:
   ```bash
   # Start base stack with DCGM monitoring
   docker-compose up -d
   
   # Optional: Add Triton for LLM serving
   docker-compose -f docker-compose.yml -f docker-compose.triton.yml up -d
   
   # Optional: Add NVIDIA NIM microservices
   docker-compose -f docker-compose.yml -f docker-compose.nim.yml up -d
   ```
4. **Access the services**:
   - UI: `http://localhost:80` (via Traefik)
   - API docs: `http://localhost/docs`
   - Grafana: `http://localhost:3000` (admin/changeme)
   - Prometheus: `http://localhost:9090`

## Detailed Documentation

- **DCGM Integration**: [README-dcgm.md](README-dcgm.md) – GPU monitoring setup.
- **TensorRT Optimization**: [README-tensorrt.md](README-tensorrt.md) – ASR acceleration guide.
- **Triton Deployment**: [README-triton.md](README-triton.md) – Model serving configuration.
- **NVIDIA NIM Integration**: [README-nim.md](README-nim.md) – Microservices deployment.
- **Performance Benchmarking**: Use `benchmark_tensorrt.py` for ASR performance tests.

## Project Structure

```
Project Aurora Echo 2.0/
├── app.py                 # FastAPI app + async pipeline wiring
├── services/              # ASR, LLM, orchestration, provider abstractions
│   ├── asr_service.py    # ASR with TensorRT support
│   ├── llm_service.py    # LLM with NIM, Triton, vLLM providers
│   ├── tensorrt_whisper.py # TensorRT backend for Whisper
│   └── providers/        # LLM and ASR providers (NIM, OpenAI, etc.)
├── docker/               # Docker configurations
│   ├── config/           # Prometheus, Grafana, Traefik configs
│   └── api.Dockerfile    # Dockerfile with TensorRT runtime
├── docker-compose.yml    # Main stack with DCGM
├── docker-compose.nim.yml # NVIDIA NIM services
├── docker-compose.triton.yml # Triton Inference Server
├── models/               # Model storage for TensorRT, Triton, NIM
├── observability.py      # Enhanced metrics with GPU correlation
├── benchmark_tensorrt.py # Performance benchmarking script
├── convert_to_tensorrt.py # TensorRT conversion script
├── README-*.md          # Detailed documentation
└── requirements.txt      # Python dependencies
```

## Performance Expectations

With NVIDIA GPU acceleration:
- **ASR Latency**: 50-100ms (2-5x faster with TensorRT)
- **LLM Latency**: 100-300ms (1.5-2x faster with Triton/NIM)
- **GPU Utilization**: Up to 80% with DCGM monitoring
- **Throughput**: 40-60 requests/second with batch processing

## License
All rights reserved.

## Version History
- **2.0**: Added NVIDIA DCGM, TensorRT, Triton, NIM integrations, enhanced observability.
- **1.0**: Initial proof-of-concept with faster-whisper, vLLM, basic observability.

## Support
For issues and contributions, refer to the documentation or contact the maintainers.
