# Async Inference & Workflow Architecture

This document describes the asynchronous inference pipeline that powers Project
Aurora Echo. The goal is to deliver partial transcripts quickly, fail over
between LLM providers, and keep the FastAPI event loop responsive even when GPU
workloads spike.

## Components

- **ASR Service (`services/asr_service.py`)**
  - Wraps `faster-whisper` and streams transcription segments through an
    `asyncio.Queue` to the main task.
  - Configurable via env vars: `ASR_MODEL_ID`, `ASR_LANGUAGE`, `ASR_BEAM_SIZE`.

- **Secure Audio Buffer (`services/audio_buffer.py`)**
  - Collects PCM chunks in memory while the client is recording.
  - Optional Fernet encryption via `AUDIO_ENCRYPTION_KEY` ensures captured audio
    is unreadable at rest.

- **Inference Orchestrator (`services/orchestrator.py`)**
  - Background worker pool that dequeues `InferenceJob` objects and invokes the
    shared processor coroutine.
  - Tuned by `INFERENCE_WORKERS`, `INFERENCE_BATCH_SIZE`, and
    `ORCHESTRATOR_BACKEND` (used for Prometheus queue depth labels).

- **LLM Service (`services/llm_service.py`)**
  - Loads provider adapters in priority order from `LLM_PROVIDER_ORDER`.
  - Supports vLLM (OpenAI-compatible endpoint), xAI Grok, OpenAI, Azure OpenAI,
    Anthropic Claude, and Google Gemini with shared retry/backoff logic.
  - Returns payloads validated by `LLMResponseModel` (Pydantic) so downstream
    code can trust the schema.

- **Workflow Integrations (`integrations/workflows.py`)**
  - After the final summary is ready, hooks can fan the content out to Slack,
    JSONL logs, or future integrations (ticketing, calendar, CRM).

- **Observability (`observability.py`)**
  - Prometheus counters/histograms capture ASR, diarisation, LLM, and end-to-end
    latency, job totals/failures, and queue depth.

## Lifecycle
1. WebSocket handler receives `{type:"stop"}` and enqueues an `InferenceJob`.
2. Worker pulls the job, increments Prometheus counters, and sends a
   `transcribing` status to the client.
3. `ASRService.stream_transcription` emits segments asynchronously; each segment
   is forwarded to the client (`partial_transcript`) and appended to the
   transcript buffer.
4. Optional diarisation labels segments with speakers for richer context.
5. `LLMService.summarize_meeting` iterates over providers until one succeeds,
   recording LLM latency and raising failure counters when needed.
6. Final payload is returned to the client, workflow hooks run, and optional TTS
   plays the summary locally.
7. Worker records end-to-end duration and completes the job.

## Operational Tips
- Ensure GPU drivers are installed so `faster-whisper` can run in CUDA mode.
- Monitor `/metrics` during load tests; increase `INFERENCE_WORKERS` if queue
  depth grows or latency spikes.
- Configure fallback providers in `.env` to avoid user-visible failures when the
  primary endpoint is down.
- When extending workflows, keep them async to avoid blocking the worker loop.

## Future Enhancements
- Streaming partial diarisation updates to the UI.
- Extending the orchestrator to use Redis / FastStream for multi-node GPU farms.
- Adding tracing (OpenTelemetry) to correlate jobs across components.
