"""NVIDIA NIM provider for ASR inference."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, Optional

import httpx

from .base import ASRProvider

LOGGER = logging.getLogger(__name__)


class NIMASRProvider(ASRProvider):
    """Provider for NVIDIA NIM ASR inference microservices."""

    name = "nim-asr"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/audio/transcriptions",
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_seconds=backoff_seconds)
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._api_key = api_key
        self._endpoint = endpoint.lstrip('/')
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio using NIM ASR."""
        url = f"{self._base_url}/{self._endpoint}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        payload: Dict[str, Any] = {
            "model": self._model,
            "audio": audio_b64,
            "response_format": "json",
        }
        if language:
            payload["language"] = language

        delay = self._backoff_seconds
        for attempt in range(1, self._max_retries + 1):
            try:
                LOGGER.debug("Sending ASR request to NIM at %s", url)
                response = self._client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                if "text" not in data:
                    raise ValueError("No text in NIM ASR response")
                return data["text"]
            except Exception as exc:
                if attempt >= self._max_retries:
                    LOGGER.exception(
                        "NIM ASR provider failed after %s attempts", attempt
                    )
                    raise
                LOGGER.warning(
                    "NIM ASR request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt, self._max_retries, exc, delay,
                )
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("Should not reach here")

    async def close(self) -> None:
        self._client.close()

    def __repr__(self) -> str:
        return f"NIMASRProvider(base_url={self._base_url}, model={self._model})"
