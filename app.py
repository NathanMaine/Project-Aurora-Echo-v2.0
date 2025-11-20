import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from observability import (
    ASR_LATENCY,
    DIARIZATION_LATENCY,
    INFERENCE_JOB_DURATION,
    INFERENCE_JOB_FAILURES,
    INFERENCE_JOBS_TOTAL,
    LLM_LATENCY,
)
from services import InferenceJob, InferenceOrchestrator, LLMService
try:
    from services import ASRService
except ImportError:
    ASRService = None
    print("Warning: ASRService not available - running in limited mode")
from services.audio_buffer import SecureAudioBuffer

load_dotenv()

LOGGER = logging.getLogger("project_aurora_echo")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_RATE = 16000
AUDIO_ENCRYPTION_KEY = os.getenv("AUDIO_ENCRYPTION_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

asr_service: Optional["ASRService"] = None
llm_service: Optional[LLMService] = None
orchestrator: Optional[InferenceOrchestrator] = None
diarization_pipeline = None
tts_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_service, llm_service, orchestrator

    try:
        # Try to create ASR service to test if dependencies are available
        test_asr = ASRService()
        asr_service = test_asr
        LOGGER.info("ASR service initialized successfully")
    except RuntimeError as e:
        asr_service = None
        LOGGER.warning("ASR service not available: %s", e)
        LOGGER.warning("Running without speech recognition capabilities")

    llm_service = LLMService()

    orchestrator = InferenceOrchestrator(
        processor=_process_job,
        workers=int(os.getenv("INFERENCE_WORKERS", "2")),
        backend_name=os.getenv("ORCHESTRATOR_BACKEND", "in-memory"),
        batch_size=int(os.getenv("INFERENCE_BATCH_SIZE", "1")),
    )
    await orchestrator.start()
    LOGGER.info("Aurora Echo startup complete")

    yield

    if orchestrator:
        await orchestrator.stop()
    if llm_service:
        await llm_service.close()
    LOGGER.info("Aurora Echo shutdown complete")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Debug: print registered routes
for route in app.routes:
    print(f"Route: {route.path} - {type(route).__name__}")


def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None and HF_TOKEN:
        try:
            from pyannote.audio import Pipeline

            LOGGER.info("Loading speaker diarization pipeline")
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
            )
            if torch.cuda.is_available():
                diarization_pipeline.to(torch.device("cuda"))
        except Exception as exc:  # pragma: no cover - runtime warning only
            LOGGER.warning("Failed to initialise diarization pipeline: %s", exc)
            diarization_pipeline = None
    return diarization_pipeline


def get_tts_engine():
    global tts_engine
    if tts_engine is None:
        import pyttsx3

        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", int(os.getenv("TTS_RATE", "180")))
    return tts_engine


async def _speak_summary(summary: str) -> None:
    if not summary or os.getenv("ENABLE_TTS", "1").lower() in {"0", "false", "no"}:
        return
    engine = get_tts_engine()

    def _run() -> None:
        engine.say(summary)
        engine.runAndWait()

    await asyncio.to_thread(_run)


async def _process_job(job: InferenceJob) -> None:
    assert asr_service is not None
    assert llm_service is not None

    job_timer = time.monotonic()
    INFERENCE_JOBS_TOTAL.inc()

    audio_np = np.frombuffer(job.audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        await job.websocket.send_json({"type": "status", "status": "transcribing", "jobId": job.job_id})

        asr_start = time.monotonic()
        transcript_segments: List[Dict[str, Any]] = []
        transcript_parts: List[str] = []

        async for segment in asr_service.stream_transcription(audio_np, job.sample_rate):
            transcript_segments.append(segment)
            text = segment.get("text", "").strip()
            if text:
                transcript_parts.append(text)
                await job.websocket.send_json(
                    {
                        "type": "partial_transcript",
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": text,
                    }
                )

        ASR_LATENCY.observe(time.monotonic() - asr_start)
        transcript_text = " ".join(transcript_parts).strip()
        if not transcript_text:
            raise ValueError("No speech detected. Please speak clearly and try again.")

        diarized_text = await _apply_diarization(transcript_segments, job.audio_data, job.sample_rate, transcript_text)

        await job.websocket.send_json({"type": "status", "status": "summarising", "jobId": job.job_id})

        llm_start = time.monotonic()
        llm_payload = await llm_service.summarize_meeting(diarized_text)
        LLM_LATENCY.observe(time.monotonic() - llm_start)
        if llm_payload is None:
            raise RuntimeError("All configured LLM providers failed to summarise the meeting.")

        result = {
            "type": "final",
            "jobId": job.job_id,
            "transcription": diarized_text,
            "summary": llm_payload.get("summary", ""),
            "actions": llm_payload.get("actions", []),
        }

        await job.websocket.send_json(result)
        await _speak_summary(result["summary"])
        INFERENCE_JOB_DURATION.observe(time.monotonic() - job_timer)
    except Exception as exc:  # pragma: no cover - runtime logging only
        INFERENCE_JOB_FAILURES.inc()
        LOGGER.exception("Inference job %s failed", job.job_id)
        await job.websocket.send_json({"type": "final", "jobId": job.job_id, "error": str(exc)})


def _bytes_to_waveform(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    waveform = torch.from_numpy(np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
    waveform = waveform.unsqueeze(0)  # shape: (1, n_samples)
    if sample_rate != MODEL_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=MODEL_RATE)
        waveform = resampler(waveform)
    return waveform


async def _apply_diarization(
    segments: List[Dict[str, Any]], audio_bytes: bytes, sample_rate: int, fallback: str
) -> str:
    if not HF_TOKEN:
        return fallback
    pipeline = get_diarization_pipeline()
    if pipeline is None or not segments:
        return fallback
    try:
        diar_start = time.monotonic()
        waveform = _bytes_to_waveform(audio_bytes, sample_rate)
        diarization = pipeline({"waveform": waveform, "sample_rate": MODEL_RATE})
        diarized_segments: List[str] = []
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            speakers: List[str] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= start < turn.end or turn.start < end <= turn.end or (start <= turn.start and end >= turn.end):
                    speakers.append(speaker)
            speaker_label = speakers[0] if speakers else "Speaker"
            diarized_segments.append(f"{speaker_label}: {text}")
        DIARIZATION_LATENCY.observe(time.monotonic() - diar_start)
        return " \n".join(diarized_segments) if diarized_segments else fallback
    except Exception as exc:  # pragma: no cover - runtime logging only
        LOGGER.warning("Diarization failed: %s", exc)
        return fallback



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    LOGGER.info("WebSocket connection attempt received")
    if orchestrator is None:
        LOGGER.warning("WebSocket rejected: orchestrator not initialized")
        await websocket.close(code=1011)
        return

    LOGGER.info("WebSocket connection accepted")
    await websocket.accept()
    buffer = SecureAudioBuffer(encryption_key=AUDIO_ENCRYPTION_KEY)
    current_sample_rate = MODEL_RATE

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    LOGGER.debug("Ignoring non-JSON text payload: %s", message["text"])
                    continue

                event_type = payload.get("type")
                if event_type == "start":
                    buffer.reset()
                    current_sample_rate = int(payload.get("sampleRate") or MODEL_RATE)
                    await websocket.send_json({"type": "status", "status": "recording"})
                elif event_type == "stop":
                    audio_bytes = buffer.to_bytes()
                    if not audio_bytes:
                        await websocket.send_json({"type": "final", "error": "No audio captured."})
                        continue
                    await websocket.send_json({"type": "status", "status": "queued"})
                    job_id = await orchestrator.enqueue(websocket, audio_bytes, current_sample_rate)
                    await websocket.send_json({"type": "status", "status": "queued", "jobId": job_id})
                else:
                    LOGGER.debug("Unknown event type received: %s", event_type)

            if "bytes" in message and message["bytes"] is not None:
                buffer.append(message["bytes"])
    except WebSocketDisconnect:
        LOGGER.info("WebSocket disconnected")
    except Exception as exc:  # pragma: no cover - runtime logging only
        LOGGER.exception("WebSocket error: %s", exc)
        await websocket.close(code=1011)


@app.get("/test")
async def test_endpoint() -> str:
    return "Aurora Echo is running"


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
