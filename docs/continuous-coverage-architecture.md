# Continuous Coverage Architecture — Design Document

**Date:** 2026-02-17
**Status:** Option A selected for implementation

---

## Problem Statement

The current Aurora Echo demo records a **fixed 5-second snippet** (`RECORDING_DURATION_MS = 5000` in `static/index.html:83`), processes it through ASR + LLM, and displays results. This is a proof-of-concept, not a meeting tool.

**Goal:** Continuous, uninterrupted meeting capture — the user clicks "Start Meeting" and Aurora Echo records, transcribes, and summarises the entire meeting in real time, accumulating a running transcript and periodically updating the summary and action items.

---

## Options Evaluated

### Option A — Continuous Stream + Server-Side Windowing (SELECTED)

**How it works:**
1. Browser records continuously (no auto-stop timer)
2. PCM audio streams via WebSocket to the server in real time (already working)
3. Server accumulates audio in a rolling buffer
4. Every N seconds (e.g. 30s), server extracts the buffer window and sends it through ASR
5. Transcription results are appended to a running transcript
6. After every M windows (e.g. every 2-3 chunks), the accumulated transcript is re-summarised via LLM
7. UI shows running transcript, live-updating summary, and growing action items table

**Pros:**
- Single WebSocket connection per client (already exists)
- Server controls the chunking — can tune window size without touching frontend
- Natural fit for the existing `SecureAudioBuffer` + `InferenceOrchestrator` architecture
- No audio gaps — server has the complete stream
- Simplest frontend change (just remove the 5s timer, add Start/Stop toggle)

**Cons:**
- Server holds more audio in memory (mitigated by `MAX_AUDIO_BUFFER_BYTES` cap)
- Periodic re-summarisation costs scale with transcript length
- Single point of failure per connection

### Option B — Client-Side Windowing + Parallel Workers

**How it works:**
1. Browser records continuously
2. Client-side JS splits audio into overlapping windows (e.g. 30s chunks with 5s overlap)
3. Each chunk is sent as an independent "job" to the server
4. Multiple orchestrator workers process chunks in parallel
5. Server stitches transcriptions together, deduplicating overlapping segments

**Pros:**
- Parallelises ASR across multiple workers
- Each chunk is independent — failure of one doesn't lose the whole stream
- Client controls timing — server is stateless per chunk

**Cons:**
- Complex overlap/deduplication logic (acoustic boundaries vs. word boundaries)
- More WebSocket messages and coordination overhead
- Client must manage its own audio buffer and timing
- Harder to test — timing-dependent edge cases

### Option C — Multiple Independent Agent Sessions

**How it works:**
1. Multiple browser tabs or API clients each record different time windows
2. A coordinator service merges results from all agents
3. Each agent is a complete Aurora Echo instance

**Pros:**
- True redundancy — one agent failure doesn't lose coverage
- Scales horizontally

**Cons:**
- Massive over-engineering for a demo
- Duplicate audio capture hardware issues
- Complex merge/dedup across independent streams
- Requires a whole new coordination layer

---

## Decision: Option A

Option A was selected because:
1. **Minimal code change** — the WebSocket streaming already works; we just need to remove the 5s timer and add periodic server-side processing
2. **Single connection** — no coordination overhead
3. **Server controls timing** — easy to tune chunk size and summarisation frequency
4. **Builds on existing architecture** — `SecureAudioBuffer`, `InferenceOrchestrator`, and the WebSocket handler all extend naturally

---

## Implementation Plan — Option A

### Phase 1: Frontend — Continuous Recording Toggle

**File:** `static/index.html`

**Changes:**
- Remove `RECORDING_DURATION_MS = 5000` auto-stop timer
- Change button to a Start/Stop toggle:
  - Idle → "Start Meeting" (green)
  - Recording → "Stop Meeting" (red, pulsing)
- On Start: send `{type: "start"}`, begin streaming PCM
- On Stop: send `{type: "stop"}`, stop mic, await final results
- Handle new message types from server:
  - `partial_transcript` — append to running transcript (already works)
  - `chunk_summary` — update summary textarea with latest summary
  - `chunk_actions` — merge into action items table (append new, don't replace)
  - `final` — show final results after Stop
- Update page description from "five-second snippet" to "meeting capture"

### Phase 2: Server — Periodic Chunk Processing

**File:** `app.py` (WebSocket handler)

**Changes:**
- Add a `CHUNK_INTERVAL_SECONDS` config (default 30s, from env `CHUNK_INTERVAL`)
- When recording is active, run a background `asyncio.Task` that:
  1. Waits `CHUNK_INTERVAL_SECONDS`
  2. Extracts current buffer contents (snapshot, not drain — keep accumulating)
  3. Sends the chunk through ASR via `asr_service.stream_transcription()`
  4. Appends transcription to session's running transcript
  5. Every N chunks (configurable), sends accumulated transcript through LLM
  6. Sends `chunk_summary` + `chunk_actions` messages to client
- On "stop" signal:
  - Cancel the periodic task
  - Process any remaining audio in the buffer
  - Send final accumulated transcript + final summary
- Track per-session state: `running_transcript`, `chunk_count`, `last_processed_offset`

### Phase 3: Audio Buffer Enhancement

**File:** `services/audio_buffer.py`

**Changes:**
- Add `snapshot(offset)` method — returns bytes from `offset` to end without clearing
- Add `byte_count` property — returns total accumulated bytes
- This allows the periodic processor to grab new audio since last chunk without losing the buffer

### Phase 4: Session State Management

**New concept (inline in `app.py`, not a new file):**
- Per-WebSocket session state dataclass:
  ```python
  @dataclass
  class MeetingSession:
      buffer: SecureAudioBuffer
      sample_rate: int
      running_transcript: str
      chunk_count: int
      last_byte_offset: int
      periodic_task: Optional[asyncio.Task]
  ```
- Created on "start", cleaned up on "stop" or disconnect

### Phase 5: Incremental Summarisation

**File:** `services/llm_service.py` or inline in `app.py`

**Changes:**
- Add method or logic for incremental summarisation:
  - First chunk: summarise from scratch
  - Subsequent chunks: send `previous_summary + new_transcript_segment` to LLM
  - Prompt: "Update this meeting summary with the new transcript segment. Keep the summary under 200 words. Merge any new action items."
- This keeps LLM cost proportional to new content, not total meeting length

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_INTERVAL` | `30` | Seconds between periodic ASR processing |
| `SUMMARIZE_EVERY_N_CHUNKS` | `2` | Run LLM summarisation every N chunks |
| `MAX_AUDIO_BUFFER_BYTES` | `104857600` (100MB) | Memory cap for audio buffer |

---

## Message Protocol (Updated)

### Client → Server
| Message | When |
|---------|------|
| `{type: "start", sampleRate: 16000}` | User clicks "Start Meeting" |
| Binary PCM frames | Continuously during recording |
| `{type: "stop"}` | User clicks "Stop Meeting" |

### Server → Client
| Message | When |
|---------|------|
| `{type: "status", status: "recording"}` | After "start" received |
| `{type: "partial_transcript", text: "..."}` | After each ASR chunk |
| `{type: "chunk_summary", summary: "...", actions: [...]}` | After periodic LLM summarisation |
| `{type: "status", status: "finalising"}` | After "stop" received |
| `{type: "final", summary: "...", actions: [...], transcription: "..."}` | Final results |

---

## Testing Strategy

1. **Unit:** Buffer `snapshot()` method with offset tracking
2. **Integration:** WebSocket sends continuous audio, verify periodic transcript updates
3. **E2E:** Browser records 2+ minutes, verify transcript grows, summary updates
4. **Edge cases:** Stop during processing, disconnect during recording, empty audio chunks, buffer overflow
