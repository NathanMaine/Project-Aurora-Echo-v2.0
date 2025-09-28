# Project Aurora Echo

AI meeting copilot that streams audio from the browser, transcribes it with
`faster-whisper`, diarises speakers with `pyannote.audio`, and produces a
structured summary using a pluggable LLM provider stack (local vLLM, Grok,
OpenAI, etc.). Everything is orchestrated asynchronously so the UI receives
status updates and partial transcripts in real time.

## Feature Highlights
- **Binary WebSocket transport** – browser sends raw PCM frames, server replies
  with status events, partial transcripts, and final JSON payloads.
- **Async inference pipeline** – audio chunks queue through an
  `InferenceOrchestrator`, `ASRService` streams transcription, and the
  `SecureAudioBuffer` protects audio in memory (optional Fernet encryption).
- **Multi-provider LLM failover** – configure provider order via environment
  variables; defaults to local vLLM (`meta-llama-3-8b-instruct`) then Grok.
- **Speaker diarisation** – opt-in via `HF_TOKEN`; falls back gracefully if the
  Hugging Face pipeline is unavailable.
- **Observability** – Prometheus counters/histograms exposed on `/metrics` for
  ASR, diarisation, and LLM latency plus queue depth and job totals.
- **Optional TTS feedback** – summaries can be spoken locally via `pyttsx3`.

## Repository Layout
```
Project Aurora Echo/
├── app.py                 # FastAPI app + async pipeline wiring
├── static/index.html      # Browser client (binary streaming + status UI)
├── services/              # ASR, orchestration, LLM provider abstractions
├── integrations/          # Hooks for Slack/webhook workflows (extensible)
├── observability.py       # Prometheus metric definitions
├── docker/                # Traefik, Prometheus, Grafana configuration
├── docker-compose.yml     # Optional stack (API + vLLM + observability)
├── docs/                  # Architecture notes and roadmap
├── requirements.txt       # Python dependencies
└── internal_future_plan.md# Private roadmap (ignored by Git)
```

## Quick Start (Local)
1. **Clone & create a virtual environment**
   ```bash
   git clone https://github.com/dentity007/Project-Aurora-Echo.git
   cd Project-Aurora-Echo
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables** (create `.env`)
   ```env
   # Primary providers
   VLLM_BASE_URL=http://localhost:8001      # optional local vLLM instance
   LLM_PROVIDER_ORDER=vllm,grok             # default failover order
   XAI_API_KEY=your-grok-token              # required if Grok is in the order

   # Optional diarisation
   HF_TOKEN=your-hf-token                   # accept pyannote terms first

   # Optional audio encryption + TTS
   AUDIO_ENCRYPTION_KEY=base64-fernet-key
   ENABLE_TTS=1
   ```

3. **Run the backend & (optionally) vLLM**
   ```bash
   # start local vLLM if you want on-device inference
   docker run --rm --gpus all -p 8001:8001 \
     vllm/vllm-openai:latest --model meta-llama-3-8b-instruct

   # start the FastAPI server
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Use the web UI** (`http://localhost:8000`)
   - Allow microphone access.
   - Click **Record & Analyze** to capture five seconds of audio.
   - Watch live transcript updates, then review the final summary & actions.

## Docker Compose Stack
The provided `docker-compose.yml` spins up:
- `meeting-assistant-api` – this FastAPI application (GPU-enabled).
- `meeting-assistant-vllm` – vLLM model runner (enable with `--profile model`).
- `meeting-assistant-proxy` – Traefik reverse proxy with HTTPS and basic auth.
- `meeting-assistant-prometheus` & `meeting-assistant-grafana` – observability.

1. Copy `.env.docker`, populate secrets (`VLLM_BASE_URL=http://vllm:8001`,
   `LLM_PROVIDER_ORDER=vllm,grok`, `XAI_API_KEY=...`, etc.).
2. Launch: `docker compose --profile model up --build`.
3. Access:
   - UI: `https://<host>/`
   - API docs: `https://<host>/docs`
   - Metrics: `https://<host>/metrics`
   - Prometheus: `http://<host>:9090`
   - Grafana: `http://<host>:3000`

## Metrics & Monitoring
`GET /metrics` returns Prometheus-formatted data. Key series:
- `meeting_assistant_inference_jobs_total`, `_failures_total`
- `meeting_assistant_asr_latency_seconds`
- `meeting_assistant_diarization_latency_seconds`
- `meeting_assistant_llm_latency_seconds`
- `meeting_assistant_inference_job_duration_seconds`
- `meeting_assistant_inference_queue_depth{backend="..."}`

## Troubleshooting
- **No LLM response** – confirm at least one provider in `LLM_PROVIDER_ORDER`
  has valid credentials and is reachable (vLLM, Grok, OpenAI, etc.).
- **Diarisation missing** – ensure `HF_TOKEN` has access to the pyannote model
  and the server has sufficient GPU memory (falls back automatically on error).
- **`WebSocket closed before opening`** – check reverse proxy/websocket headers
  and that the backend is running on port 8000.
- **High latency** – monitor `/metrics`; adjust `INFERENCE_WORKERS` and
  `INFERENCE_BATCH_SIZE` to match GPU capacity.
- **Disable TTS** – set `ENABLE_TTS=0` or unset the variable.

## License
All rights reserved.
