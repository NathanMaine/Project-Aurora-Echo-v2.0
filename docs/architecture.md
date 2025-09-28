# Architecture Overview

Project Aurora Echo couples a FastAPI backend with a lightweight browser client
that streams PCM audio over WebSockets. Audio ingestion, transcription, and LLM
summarisation are decoupled via an asynchronous orchestration layer.

## Request Flow
1. The browser sends `{type:"start"}` to open a session, then streams `Int16`
   PCM frames. After the five-second capture window it sends `{type:"stop"}`.
2. `SecureAudioBuffer` accumulates PCM chunks (optionally encrypting them with a
   Fernet key) until the session ends.
3. An `InferenceOrchestrator` enqueues the job and dispatches it to worker tasks.
4. `ASRService` (backed by `faster-whisper`) streams transcription segments. Each
   segment is forwarded to the client as a `partial_transcript` event while ASR
   latency is recorded in Prometheus.
5. If `HF_TOKEN` is configured, the diarisation pipeline annotates segments with
   speaker labels. Failures fall back to the raw transcript and are logged.
6. `LLMService` calls providers in priority order (vLLM → Grok → OpenAI/Azure →
   Anthropic → Gemini), returning structured JSON validated by Pydantic models.
7. Final results (`summary`, `actions`) are sent back to the client as a `final`
   event and Prometheus job counters/histograms are updated. Optional TTS plays
   the summary locally via `pyttsx3`.

## Key Components
- **`app.py`** – FastAPI app, WebSocket handler, startup/shutdown wiring, and
  Prometheus endpoint.
- **`services/asr_service.py`** – `faster-whisper` wrapper supporting async
  streaming with configurable model size and device.
- **`services/orchestrator.py`** – asyncio worker queue with batching, backend
  labelling, and graceful shutdown logic.
- **`services/llm_service.py`** – provider registry + retry/backoff logic calling
  vLLM, Grok, OpenAI, Azure OpenAI, Anthropic Claude, and Gemini adapters.
- **`services/audio_buffer.py`** – in-memory PCM accumulator with optional
  Fernet encryption/decryption.
- **`integrations/workflows.py`** – post-meeting hooks (Slack webhook + JSONL log)
  ready for expansion.
- **`observability.py`** – Prometheus counters/histograms shared across modules.

## Deployment Options
- **Local development** – run `uvicorn app:app --reload` and optionally launch a
  local vLLM container on port 8001.
- **Docker Compose** – builds the API image, Traefik reverse proxy, observability
  stack, and a managed vLLM service (via profile `model`).

## Extensibility Notes
- Increase `INFERENCE_WORKERS` and `INFERENCE_BATCH_SIZE` to improve throughput.
- Swap `LLM_PROVIDER_ORDER` at runtime to prioritise different providers.
- Extend `integrations/workflows.py` for ticketing, calendar updates, etc.
- Expose additional metrics or tracing by augmenting `observability.py`.

Refer to `docs/async-inference-service.md` and
`docs/streaming-transport-upgrade.md` for deeper dives into the inference and
transport layers.
