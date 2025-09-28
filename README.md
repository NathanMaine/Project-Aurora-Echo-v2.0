# Project Aurora Echo

Real-time meeting aide that records a short audio clip in your browser, transcribes it with OpenAI Whisper, and asks xAI Grok to produce a summary and action items. The current implementation is intentionally simple while we stage larger upgrades.

## Current Capabilities
- Five-second microphone capture in the browser; audio is sent as a base64-encoded WAV payload over WebSocket.
- Whisper `base` model (CUDA if available) handles transcription.
- Optional speaker diarisation via `pyannote.audio` when `HF_TOKEN` is provided.
- Grok (`grok-3`) generates a JSON summary and action list.
- Optional desktop text-to-speech feedback using `pyttsx3`.

## Roadmap Snapshot
We are actively working on the following enhancements (tracked in `internal_future_plan.md`):
- Async inference orchestrator with `faster-whisper` streaming and binary WebSocket transport.
- Provider failover using the abstractions under `services/providers/` (local vLLM first, Grok/OpenAI/etc. as fallbacks).
- Prometheus metrics, richer observability, and workflow integrations.
- Modernised front-end with session history and export tools.

Public documentation will expand as each feature ships.

## Project Layout
```
Project Aurora Echo/
├── app.py                 # Current FastAPI application
├── static/index.html      # Minimal browser client for recording & results
├── services/              # Upcoming service abstractions (not wired in yet)
├── integrations/          # Future workflow hooks (Slack, logs, etc.)
├── docs/                  # Design notes and upgrade plans
├── docker/                # Container configuration and reverse proxy
├── docker-compose.yml     # Optional stack with Traefik, Prometheus, vLLM stub
└── requirements.txt       # Python dependencies for the current app
```

## Getting Started

### 1. Clone & Create Environment
```bash
git clone https://github.com/dentity007/Project-Aurora-Echo.git
cd Project-Aurora-Echo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Secrets
Create a `.env` file in the project root:
```env
XAI_API_KEY=your-xai-api-key
# Optional: enable diarisation if you accepted the pyannote terms on Hugging Face
HF_TOKEN=your-hf-token
```

### 3. Run the App
```bash
python app.py
# or
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your browser, allow microphone access, and click **Record & Analyze (5 seconds)**.

## Docker Notes
- `docker-compose.yml` builds the FastAPI container and can launch Traefik/Prometheus.
- A `vllm` service is included for future work but is **not** consumed by the current app. Do not expect responses from it yet.
- Provide secrets via `.env.docker` before running `docker compose up --build`.

## Troubleshooting
- **`API_KEY not found`**: ensure `XAI_API_KEY` is set in your environment or `.env`.
- **No speaker labels**: set `HF_TOKEN` and verify access to `pyannote/speaker-diarization-3.1`.
- **Audio errors on macOS/Windows**: install PortAudio development headers; see `pyaudio` documentation.
- **Grok timeouts**: responses take ~20s; adjust `timeout` in `app.py` if needed.

## License
All rights reserved.
