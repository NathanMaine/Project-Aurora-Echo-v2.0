# Autonomous Build Instructions

This document guides an automated AI engineer ("builder agent") through setting up,
verifying, and extending the Real-Time AI Meeting Assistant. Follow the numbered
checklists in order. Treat unchecked items as TODOs for subsequent iterations.

## 1. Environment Provisioning
1. Detect OS/hardware; confirm Windows 11 Pro 64 on Lenovo ThinkPad P16 Gen 2.
2. Ensure NVIDIA RTX 5000 Ada drivers and CUDA runtime are installed (`nvidia-smi`).
3. Install Python 3.10/3.11 (64-bit), Git, Visual Studio Build Tools (C++), and
   upgrade `pip`, `setuptools`, `wheel`.
4. Create a virtual environment (`python -m venv .venv`) and run `pip install -r requirements.txt`.
5. Install and launch vLLM (default: `meta-llama-3-8b-instruct`) on port 8001 or
   configure an alternate provider through `.env`.
6. Populate `.env` with mandatory/optional values (see README & Setup Guide).
7. Tune queue settings with `INFERENCE_WORKERS`, `INFERENCE_BATCH_SIZE`, and `ORCHESTRATOR_BACKEND` to match GPU capacity.
8. Build Docker images (`docker compose --profile model up --build -d`) when testing LAN deployments; ensure `nvidia-container-toolkit` is configured.
9. Load API credentials from a secret manager (Credential Manager, Vault, SOPS) rather than committing them to source control.

## 2. Sanity Checks
1. `python -c "import torch; print(torch.cuda.is_available())"` → must print `True`.
2. `uvicorn app:app --reload` → server should start without model-loading errors.
3. Browser WS test: record a sample; verify `partial_transcript` frames in DevTools.
4. Confirm LLM response arrives from vLLM (watch logs). If failure, ensure Grok
   fallback works when `XAI_API_KEY` is set.
5. Scrape `http://localhost:8000/metrics` and ensure Prometheus counters increment (queue depth, job totals, latency histograms).

## 3. Multi-Provider LLM Support Roadmap
1. Current capability: local vLLM primary, Grok fallback with provider ordering (`LLM_PROVIDER_ORDER`).
2. Next providers to implement:
   - OpenAI / Azure OpenAI (GPT-4o mini)
   - Anthropic Claude 3.x
   - Google Gemini 1.5 Pro
3. Approach:
   - Implement provider adapters under `services/providers/` (DONE for vLLM + Grok).
   - Add additional provider adapters with shared retry/backoff + pydantic validation.
   - Extend UI to surface provider selection (future React client).

## 4. High-Impact Upgrade Backlog
Reference for prioritisation (see user requirements):
- Real-time streaming (DONE).
- GPU-centric inference pipeline (PARTIAL: models preload on startup; next add batched diarization, NeMo experiments, streaming whisper-large-v3).
- Structured orchestration (PARTIAL: in-process asyncio worker queue + batching controls live; next step is external dispatcher via FastStream/Redis).
- httpx.AsyncClient + pydantic validation for provider calls (DONE for vLLM + Grok; extend to future providers).
- Secure in-memory audio, optional encryption for persisted buffers (DONE via `SecureAudioBuffer` + `AUDIO_ENCRYPTION_KEY`).
- Meeting intelligence workflows (agenda ingest, Jira/email, RAG memory) (TODO).
- Multi-modal capture (slides OCR, screen analysis) (TODO).
- Neural TTS and UX cues (TODO).
- React/TypeScript front-end with heat maps, inline editing, exports (TODO).
- Offline/Electron support and voice command UX (TODO).
- Observability & secrets (PARTIAL: Prometheus metrics exposed; next integrate tracing, secrets vault, automated redaction).

## 5. Documentation Obligations
1. Update `README.md`, setup guides, and architecture docs after every
   substantive code change.
2. Append new operational notes to `docs/async-inference-service.md` and
   `docs/streaming-transport-upgrade.md` when altering transport/inference logic.
3. For new workflows or providers, add how-to sections and env var tables.
4. Maintain this document as the authoritative TODO list for automated agents.

## 6. Testing & Validation
1. Add pytest-based integration tests using prerecorded audio fixtures.
2. Validate LLM JSON responses against pydantic schemas.
3. Create smoke scripts for Slack/webhook workflows (dry-run mode).
4. Capture GPU utilisation metrics during load tests; ensure no memory leaks.

## 7. Deployment Targets
1. Local single-user (current): uvicorn + vLLM.
2. Planned: Docker compose stack (FastAPI, Redis/Message queue, vLLM, vector DB).
3. Future: Kubernetes with GPU scheduling, auto-scaling inference workers.

Keep this checklist synchronized with implementation progress so future AI agents
can continue the build with minimal context switching.
