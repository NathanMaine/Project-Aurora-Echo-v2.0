# Binary WebSocket Audio Streaming Upgrade (Planned)

> **Status:** Design draft. The live client/server still use base64 WAV payloads
> and the legacy WebSocket handler in `app.py`. Treat this file as guidance for
> the future transport upgrade.

This document explains the intended switch from base64 WAV transport to binary
PCM streaming, how the new client–server contract will work, and how to
troubleshoot the updated pipeline once it is implemented. Use it as a reference
when preparing the upcoming release of the real-time meeting assistant.

## What Changed
- **Binary frames instead of base64 text**: The browser now sends raw `Int16` PCM buffers over the WebSocket, removing temp-file writes and base64 expansion (`static/index.html:81-256`).
- **Explicit session control messages**: Each recording session is wrapped with JSON `{type: "start", sampleRate}` and `{type: "stop"}` directives so the server knows when to reset buffers (`static/index.html:209-274`).
- **In-memory audio processing**: The backend consumes PCM buffers directly, resamples in-memory when necessary, and feeds faster-whisper without touching disk (`app.py:96-147`). Optional Fernet encryption protects audio buffers while queued.
- **Unified WebSocket handler**: The server accumulates binary frames, reacts to control messages, streams partial transcript/status updates, and speaks the summary once processing completes (`app.py:228-283`).

## Runtime Flow
1. User clicks **Record & Analyze**. The UI ensures the socket is open and sends `{type: "start", sampleRate: 16000}`.
2. A `ScriptProcessorNode` emits PCM frames. Each frame is clamped to `[-1, 1]`, converted to signed 16-bit little-endian integers, and dispatched via `ws.send(pcmBuffer)`.
3. After 5 seconds the UI disconnects audio nodes, stops microphone tracks, and sends `{type: "stop"}`.
4. The server concatenates the received frames, reconstructs a waveform tensor, performs optional resampling, runs faster-whisper transcription, applies diarization (if the pipeline is available), and invokes the LLM service (vLLM primary, Grok fallback).
5. Transcription, summary, and action items are returned as JSON. The backend speaks the summary via `pyttsx3` before clearing the buffer for the next recording and firing workflow integrations.

## How to Verify Without Local Execution
- **Code inspection**: Confirm `static/index.html` sends binary buffers (look for `ws.send(pcmBuffer)` at `static/index.html:255`). Ensure the `start`/`stop` messages exist (`static/index.html:209-274`) and that status/partial updates are handled.
- **Dependency check**: Ensure `faster-whisper`, `httpx`, and `cryptography` are installed (see `requirements.txt`).
- **Backend contract**: Verify `process_audio_session` accepts `audio_data`/`sample_rate` (`app.py:91-147`), and that the WebSocket loop updates the secure buffer + `current_sample_rate` before enqueuing jobs (`app.py:228-271`).
- **Environment variables**: `VLLM_*` configure the primary LLM, `XAI_API_KEY`/`OPENAI_API_KEY`/`AZURE_OPENAI_*`/`ANTHROPIC_API_KEY`/`GOOGLE_GEMINI_API_KEY` extend provider coverage, `LLM_PROVIDER_ORDER` sets precedence, `HF_TOKEN` enables diarization, `NEMO_DIARIZATION_ENABLED` toggles future NeMo support, `AUDIO_ENCRYPTION_KEY` encrypts buffers, and `INFERENCE_WORKERS`/`INFERENCE_BATCH_SIZE`/`ORCHESTRATOR_BACKEND` tune queue behaviour (`app.py:28-40`).

## Troubleshooting Guide

### WebSocket never opens
- **Symptom**: UI shows alert “Web Audio API is not supported” or logs repeated connection attempts.
- **Checks**:
  - Use a modern Chromium/Firefox build; Safari < 14 lacks `ScriptProcessorNode` for secure contexts.
  - Ensure the app is served as `http://localhost:8000`; mixed-content restrictions can block WS upgrades.

### No audio reaches the server
- **Symptom**: Backend logs stay silent or only show `Starting new audio capture...` without byte counts.
- **Checks**:
  - Confirm microphone permission was granted in the browser; blocked permissions prevent `onaudioprocess` from firing.
  - Inspect developer tools → Network → WS frames; binary frames should appear after `start`.
  - Look for `navigator.mediaDevices.getSupportedConstraints().sampleRate` support. If undefined, the constraint is skipped and actual device rate is used; resampling handles mismatches.

### Processing error after `stop`
- **Symptom**: UI receives `{"error": "No speech detected..."}` or a generic processing error.
- **Checks**:
  - Review server logs for traceback; missing CUDA drivers or Whisper weight issues surface here.
  - Ensure the total byte count is non-zero (`Processing accumulated audio buffer of X bytes`). If zero, the microphone stream emitted silence; test hardware with `arecord` or a browser audio test.
  - When resampling, `torchaudio` requires `sox` backend; make sure `torchaudio` is compiled for your Python/OS combo.

### Diarization disabled unexpectedly
- **Symptom**: Output lacks speaker tags though `HF_TOKEN` is present.
- **Checks**:
  - Confirm the Hugging Face token has access to `pyannote/speaker-diarization-3.1`.
  - Watch for GPU memory pressure; large Whisper models + diarization may exceed VRAM. Falling back to CPU increases latency but preserves functionality.

### Text-to-speech blocks UI updates
- **Symptom**: Summary appears late or UI feels frozen after each turn.
- **Checks**:
  - `pyttsx3` runs synchronously. If responsiveness is critical, move the TTS call onto a background thread or disable it when running headless (e.g., set an env flag checked before `engine.say`).

## Operational Tips
- **Resource preloading**: Warm the Whisper and diarization models at startup to front-load GPU memory allocation and avoid first-request stalls.
- **Logging**: Use `uvicorn --log-level debug` during testing to monitor `audio_buffer` growth and control messages.
- **Extending the protocol**: When adding streaming transcription, reuse the same binary transport and emit interim transcripts from the server to the client `ws.send_json({type: "partial", text, timestamp})`.

Keep this document alongside deployment notes so anyone maintaining the assistant understands the transport upgrade and failure modes.
