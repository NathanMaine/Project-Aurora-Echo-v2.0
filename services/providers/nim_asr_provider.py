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

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/audio/transcriptions",
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(max_retries, backoff_seconds)
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint.lstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def _make_request(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make request to NIM ASR endpoint."""
        url = f"{self.base_url}/{self.endpoint}"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        payload = {
            "model": self.model,
            "audio": audio_b64,
            "response_format": "json",
        }
        if language:
            payload["language"] = language
        # Add any extra parameters
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value

        LOGGER.debug("Sending ASR request to NIM at %s", url)
        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio using NIM ASR."""
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                data = self._make_request(
                    audio_data=audio_data,
                    language=language,
                    **kwargs,
                )
                latency = time.time() - start_time

                # Extract transcription text
                if "text" not in data:
                    raise ValueError("No text in NIM ASR response")
                transcription = data["text"]
                LOGGER.debug("NIM ASR transcription completed in %.2f seconds", latency)
                return transcription
            except Exception as e:
                LOGGER.warning(
                    "Attempt %d/%d failed for NIM ASR provider: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_seconds * (2 ** attempt))
        raise RuntimeError("Should not reach here")

    def __repr__(self) -> str:
        return f"NIMASRProvider(base_url={self.base_url}, model={self.model})"
