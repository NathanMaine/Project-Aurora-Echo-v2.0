# Current Architecture Overview

Project Aurora Echo is a FastAPI application that serves a lightweight browser client. The flow today is intentionally linear while we stage larger asynchronous upgrades.

## High-Level Flow
1. User clicks **Record & Analyze** on the web UI.
2. Browser records five seconds of mono audio, converts it to a base64 WAV string, and sends it over a WebSocket text frame (`static/index.html`).
3. The server saves the audio to a temporary file, loads the Whisper `base` model (CUDA if available), and transcribes the clip (`app.py`).
4. If a `HF_TOKEN` is configured, pyannote diarisation annotates segments with speaker labels.
5. The diarised transcript is sent to xAI Grok (`grok-3`) with a prompt that asks for a JSON summary plus action items.
6. The JSON response becomes the payload returned to the WebSocket client and, optionally, voiced via `pyttsx3`.

## Components in Use
- **FastAPI + WebSockets** – API surface and transport.
- **Whisper (openai-whisper)** – transcription of the short audio clip.
- **pyannote.audio** – optional speaker diarisation when the token is provided.
- **xAI Grok API** – meeting summary and action item extraction.
- **pyttsx3** – optional text-to-speech feedback on the host machine.

## Components Under Development
The repository already contains scaffolding for the next iteration (see `services/` and `docs/*upgrade.md`). These modules are design work and are not connected to the runtime yet.

- `services/asr_service.py`: async streaming ASR via `faster-whisper`.
- `services/llm_service.py`: multi-provider LLM orchestration with retries.
- `services/orchestrator.py`: background inference queue and batching support.
- `integrations/workflows.py`: hooks for Slack webhooks and audit logs.
- `observability.py`: Prometheus metrics definitions.

These will replace the in-process workflow once the async pipeline is implemented.

## Deployment Today
- Run locally with `python app.py` or `uvicorn` for development.
- Docker Compose builds the API container and optionally starts Traefik, Prometheus, and a placeholder vLLM service (future work).

## Known Limitations
- Audio upload relies on temporary files and base64 transport, which adds overhead.
- Only Grok is called; provider failover is not enabled yet.
- No Prometheus metrics are published by the live app.
- TTS runs synchronously and can block the event loop during long summaries.

Refer to `internal_future_plan.md` for the private roadmap and `docs/async-inference-service.md` for the planned async design (labelled as future work).
