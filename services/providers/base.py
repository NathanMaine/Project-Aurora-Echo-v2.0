"""Abstract base class for LLM and ASR providers."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

import httpx
from pydantic import ValidationError

from services.providers.models import LLMResponseModel

LOGGER = logging.getLogger(__name__)


class LLMProvider(abc.ABC):
    """Interface for LLM backends."""

    name: str

    def __init__(self, *, max_retries: int = 3, backoff_seconds: float = 1.0) -> None:
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def backoff_seconds(self) -> float:
        return self._backoff_seconds

    @abc.abstractmethod
    async def summarize(self, transcript: str) -> LLMResponseModel:
        """Produce a structured meeting summary.

        Should raise an exception on failure so the caller can try the next provider.
        """

    async def _run_with_retry(
        self,
        func: Callable[[], Awaitable[LLMResponseModel]],
    ) -> LLMResponseModel:
        """Execute *func* with exponential backoff on transient failures."""
        delay = self._backoff_seconds
        attempt = 0
        while True:
            try:
                return await func()
            except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                attempt += 1
                if attempt >= self._max_retries:
                    LOGGER.exception(
                        "%s provider failed after %s attempts", self.name, attempt
                    )
                    raise
                LOGGER.warning(
                    "%s request failed (attempt %s/%s): %s; retrying in %.1fs",
                    self.name, attempt, self._max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def close(self) -> None:
        """Release any resources held by the provider."""


class ASRProvider(abc.ABC):
    """Interface for ASR backends."""

    name: str

    def __init__(self, *, max_retries: int = 3, backoff_seconds: float = 1.0) -> None:
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def backoff_seconds(self) -> float:
        return self._backoff_seconds

    @abc.abstractmethod
    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text.

        Should raise an exception on failure so the caller can try the next provider.
        """

    async def close(self) -> None:
        """Release any resources held by the provider."""
