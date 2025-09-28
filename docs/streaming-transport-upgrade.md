# Binary WebSocket Audio Streaming

The browser and server now communicate using binary PCM frames instead of base64
WAV payloads. This reduces bandwidth, eliminates temporary files, and enables
real-time status updates while the inference pipeline runs.

## Protocol Overview
- **Start** – client sends `{type:"start", sampleRate: 16000}` to reset buffers.
- **Streaming** – each PCM frame is encoded as `Int16Array` and sent as a binary
  WebSocket frame (`ws.send(int16Array.buffer)`).
- **Stop** – client sends `{type:"stop"}` to indicate the end of the capture.
- **Server events** – backend emits JSON messages:
  - `{"type":"status","status":"queued"}` etc. for lifecycle updates.
  - `{"type":"partial_transcript", ...}` for each ASR segment.
  - `{"type":"final", ...}` containing transcription, summary, actions, or
    an error message.

## Server-Side Handling (`app.py`)
1. `SecureAudioBuffer` appends raw PCM frames as they arrive.
2. On `stop`, the buffer decrypts/concatenates audio and enqueues an
   `InferenceJob` with the current sample rate.
3. Worker tasks feed the PCM data directly to `faster-whisper`, no disk IO.
4. Transcripts, diarisation, and LLM summaries are streamed back to the client.

## Client-Side Handling (`static/index.html`)
1. Uses `AudioContext` + `ScriptProcessorNode` to capture mono audio.
2. Resamples to 16 kHz in JavaScript when the hardware sample rate differs.
3. Converts float samples to signed 16-bit integers and sends them over the socket.
4. Handles server status updates, appending partial transcripts and rendering the
   final summary/action table.

## Troubleshooting
- **WebSocket stays disconnected** – ensure TLS termination/proxies forward
  upgrade headers; check Traefik logs if running in Docker.
- **Audio is silent** – confirm microphone permissions and that `stop` is sent
  (pressing the button fires automatically after five seconds).
- **Distorted audio** – verify resampling math or lower the buffer size (4096)
  if the browser stutters on slower hardware.

## Future Ideas
- Switch to `AudioWorklet` for lower latency capture.
- Accept live streaming sessions (no fixed 5-second window).
- Add adaptive chunk sizing based on network conditions.
