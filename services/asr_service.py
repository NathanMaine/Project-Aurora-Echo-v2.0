"""Async ASR service backed by faster-whisper with TensorRT fallback."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Dict, Optional

import numpy as np
import torch

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    FasterWhisperModel = None

try:
    from .tensorrt_whisper import WhisperModel as TensorRTWhisperModel
try:
    from .providers.nim_asr_provider import NIMASRProvider
    NIM_ASR_AVAILABLE = True
except ImportError:
    NIM_ASR_AVAILABLE = False
    NIMASRProvider = None
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    TensorRTWhisperModel = None

LOGGER = logging.getLogger(__name__)


def get_whisper_model(
    model_size: str,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    **kwargs,


def get_asr_provider(
    model_size: str,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    **kwargs,
):
    """Factory function returning appropriate ASR provider."""
    import os
    from pathlib import Path
    
    # Check if NIM ASR is enabled via environment
    nim_asr_base_url = os.getenv("NIM_ASR_BASE_URL")
    if NIM_ASR_AVAILABLE and nim_asr_base_url:
        LOGGER.info("Using NIM ASR backend")
        return NIMASRProvider(
            base_url=nim_asr_base_url,
            model=model_size,
            api_key=os.getenv("NIM_ASR_API_KEY"),
            endpoint=os.getenv("NIM_ASR_ENDPOINT", "/v1/audio/transcriptions"),
            max_retries=int(os.getenv("NIM_ASR_MAX_RETRIES", "3")),
            backoff_seconds=float(os.getenv("NIM_ASR_BACKOFF_SECONDS", "1.0")),
            timeout=float(os.getenv("NIM_ASR_TIMEOUT", "60.0")),
        )
    
    # Fallback to local Whisper model
    return get_whisper_model(model_size, device, compute_type, **kwargs)
):
    """Factory function returning appropriate Whisper model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = compute_type or ("auto" if device == "cuda" else "float32")
    
    # Check if TensorRT engine exists and CUDA is available
    if TENSORRT_AVAILABLE and device == "cuda":
        import os
        from pathlib import Path
        engine_path = f"./models/whisper-{model_size}-{compute_type if compute_type != 'auto' else 'float16'}.plan"
        if Path(engine_path).exists():
            LOGGER.info("Using TensorRT backend")
            return TensorRTWhisperModel(
                model_size=model_size,
                engine_path=engine_path,
                device=device,
                compute_type=compute_type if compute_type != "auto" else "float16",
            )
    
    # Fallback to faster-whisper
    if FASTER_WHISPER_AVAILABLE:
        LOGGER.info("Using faster-whisper backend")
        return FasterWhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            **kwargs,
        )
    else:
        raise RuntimeError(
            "No ASR backend available. Install faster-whisper or TensorRT dependencies."
        )


class ASRService:
    """Wraps Whisper model to expose an async streaming API."""

    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        language: Optional[str] = None,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._compute_type = compute_type or ("auto" if self._device == "cuda" else "float32")
        LOGGER.info(
            "Loading Whisper model %s on %s (%s)",
            model_size,
            self._device,
            self._compute_type,
        )
        self._model = get_asr_provider(
            model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        self._beam_size = beam_size
        self._language = language

    async def stream_transcription(
        self, audio: np.ndarray, sample_rate: int
    ) -> AsyncIterator[Dict[str, float | str]]:
        """Yield transcription segments asynchronously as they become available."""

        queue: asyncio.Queue[Optional[Dict[str, float | str]]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker() -> None:
            try:
                segments, info = self._model.transcribe(
                    audio,
                    beam_size=self._beam_size,
                    language=self._language,
                    vad_filter=True,
                    chunk_length=15,
                    temperature=[0.0],
                    sample_rate=sample_rate,
                )
                LOGGER.debug(
                    "ASR info: language=%s, duration=%.2fs",
                    info.language,
                    info.duration,
                )
                for segment in segments:
                    payload = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "avg_logprob": getattr(segment, "avg_logprob", 0.0),
                        "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                    }
                    loop.call_soon_threadsafe(queue.put_nowait, payload)
            except Exception as exc:  # pragma: no cover - runtime logging only
                LOGGER.exception("ASR worker failed: %s", exc)
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _worker)

        while True:
            item = await queue.get()
            if item is None:
                break
            if "error" in item:
                raise RuntimeError(item["error"])
            yield item
