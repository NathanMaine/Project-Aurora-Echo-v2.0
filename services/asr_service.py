"""Async ASR service backed by faster-whisper."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Dict, Optional

import numpy as np
import torch

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

LOGGER = logging.getLogger(__name__)


class ASRService:
    """Wraps faster-whisper to expose an async streaming API."""

    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        language: Optional[str] = None,
    ) -> None:
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is required for ASRService. Install it via: pip install faster-whisper"
            )
        
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._compute_type = compute_type or ("auto" if self._device == "cuda" else "float32")
        LOGGER.info(
            "Loading faster-whisper model %s on %s (%s)",
            model_size,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
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
