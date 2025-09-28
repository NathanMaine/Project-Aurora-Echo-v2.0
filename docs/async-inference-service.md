# Async Inference & Workflow Architecture (Planned)

> **Status:** Design draft. The workflow described below is not wired into the
> current FastAPI app yet. See `app.py` and `static/index.html` for the
> implementation that ships today, and track the private roadmap in
> `internal_future_plan.md`.

This guide explains the upcoming asynchronous inference pipeline we intend to
build around `faster-whisper` (ASR) and a local vLLM deployment for LLM
responses. It also describes how partial results would reach the UI and how
post-meeting workflows could be triggered once the redesign lands.

## Components

- **ASR Service (`services/asr_service.py`)**
  - Loads `faster-whisper` on the RTX GPU and streams segments through an
    `asyncio.Queue` so the WebSocket can forward partial transcripts in near
    real time.
  - Environment tweaks:
    - `ASR_MODEL_ID` (default `medium`)
    - `ASR_LANGUAGE` (optional explicit language)

- **Inference Orchestrator (`services/orchestrator.py`)**
  - Maintains a background worker queue so inference jobs run off the hot path.
  - Configurable via `INFERENCE_WORKERS`, `INFERENCE_BATCH_SIZE`, and `ORCHESTRATOR_BACKEND` (label for metrics / future external queues).

- **LLM Service (`services/llm_service.py`)**
  - Connects to a vLLM server that exposes the OpenAI-compatible
    `/v1/chat/completions` endpoint. Uses `httpx.AsyncClient` and enforces JSON
    output via `response_format`.
  - Environment:
    - `VLLM_BASE_URL` (default `http://localhost:8001`)
    - `VLLM_MODEL_ID` (default `meta-llama-3-8b-instruct`)
    - `VLLM_COMPLETIONS_ENDPOINT` (default `/v1/chat/completions`)
    - `VLLM_API_KEY` (optional bearer token if the endpoint is secured)
  - Falls back to xAI Grok if `XAI_API_KEY` is set and the local server fails.
  - **Provider selection & retries**: control order and robustness with
    `LLM_PROVIDER_ORDER`, `LLM_MAX_RETRIES`, `LLM_BACKOFF_SECONDS`.
  - Additional providers: `OPENAI_API_KEY`, `AZURE_OPENAI_*`, `ANTHROPIC_API_KEY`,
    and `GOOGLE_GEMINI_API_KEY` enable the respective adapters.
  - **Multi-provider roadmap**: extend this class with additional adapters (OpenAI, Azure OpenAI, Anthropic Claude, Gemini) and select providers at runtime via environment or user preferences.

- **Secure Audio Buffer (`services/audio_buffer.py`)**
  - Keeps PCM chunks in memory and optionally encrypts them with a Fernet key
    supplied via `AUDIO_ENCRYPTION_KEY`.
- **NeMo toggle**
  - Set `NEMO_DIARIZATION_ENABLED=true` to experiment with future NeMo diarization support (currently falls back to pyannote with warnings if the toolkit is unavailable).

- **Workflow hooks (`integrations/workflows.py`)**
  - Currently supports Slack webhooks and JSONL audit logging. Extend this file
    for ticketing systems, calendar updates, etc.
  - Environment:
    - `MEETING_SLACK_WEBHOOK_URL` for Slack notifications
    - `MEETING_LOG_PATH` for a newline-delimited JSON audit trail

- **WebSocket loop (`app.py`)**
  - Streams PCM chunks, encrypts/stashes them in a secure buffer, and enqueues
    jobs for the orchestrator.
  - Emits `partial_transcript` messages as soon as the ASR worker produces
    segments and relays status updates (`Queued`, `Processing audio`, etc.).
  - Final responses use the `{"type": "final"}` envelope and include any
    errors, making it easy for the frontend to distinguish terminal messages.

## Startup & Shutdown

- The startup hook preloads faster-whisper, diarization, and LLM clients, then
  starts the orchestrator workers.
- Shutdown gracefully stops workers and closes provider clients to release
  sockets/threads.

## GPU Utilisation Notes

- `faster-whisper` selects `compute_type="auto"` on CUDA and `float32` on CPU.
  Override via `ASR_COMPUTE_TYPE` if you need explicit `float16`/`bfloat16`
  optimisation.
- Ensure the NVIDIA driver exposes compute capability ≥ 7.0 for best bfloat16
  performance with RTX 5000 Ada.
- Run `nvidia-smi` and monitor utilisation; the ASR thread uses the default
  executor, so consider pinning it to a dedicated `ThreadPoolExecutor` if you
  want stricter scheduling.

## Troubleshooting

| Symptom | Likely Cause | Remediation |
| ------- | ------------ | ----------- |
| `ASR error: ... is not installed` | `faster-whisper` wheel missing | Re-run `pip install -r requirements.txt` in the GPU-enabled environment. |
| Partial transcripts never arrive | vLLM/ASR blocking the event loop | Confirm the `partial_transcript` messages appear in DevTools → WS. If not, check server logs for ASR exceptions. |
| Final message reports `Failed to get LLM response.` | vLLM endpoint down or wrong URL | Verify `VLLM_BASE_URL` and that `curl` to `/v1/models` succeeds. The server will fall back to Grok only if `XAI_API_KEY` is configured. |
| Slack notifications missing | Webhook not set or HTTP error | Set `MEETING_SLACK_WEBHOOK_URL`. Review server logs for `Slack notification failed`. |
| Speaker labels absent | `HF_TOKEN` unset or diarization GPU OOM | Provide a Hugging Face token and consider reducing ASR model size if VRAM is tight. |
| `Audio decryption failed` | Incorrect `AUDIO_ENCRYPTION_KEY` | Regenerate the key (Fernet) and restart the server so buffers can be decrypted. |

## Extending Workflows & Providers

- Add calendar scheduling: create an async function in `integrations/workflows`
  that calls your calendar API and trigger it from `run_post_meeting_workflows`.
- Ticketing systems: convert the `actions` list into issue payloads (Jira,
  Linear, etc.) and batch them with `httpx.AsyncClient`.
- Additional LLM providers: create subclasses of `LLMService` or inject a
  strategy object that handles provider-specific payloads and schema validation.

## External Orchestration (Roadmap)
- Increase `INFERENCE_WORKERS` and `INFERENCE_BATCH_SIZE` to scale within a
  single process.
- `ORCHESTRATOR_BACKEND` currently labels metrics; future versions can map this
  to FastStream/Redis-based dispatchers that serialize `InferenceJob` payloads
  for GPU worker fleets.
- Remote workers can reuse `process_audio_session` and push `final` WebSocket
  payloads back via a callback channel (HTTP/WebSocket/FastStream).

## Metrics & Observability
- Prometheus metrics are exposed at `/metrics` (see `observability.py`).
- Key series: `meeting_assistant_inference_queue_depth`, latency histograms for ASR/LLM/diarization, job totals/failures.
- Scrape the endpoint with Prometheus/Grafana and alert on sustained queue growth or elevated failure counts.

Keep this file aligned with future service changes so operators understand how
to configure and debug the assistant.
