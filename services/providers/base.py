"""Abstract base class for LLM providers."""

from __future__ import annotations

import abc
from typing import Optional

from services.providers.models import LLMResponseModel


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

    async def close(self) -> None:
        """Release any resources held by the provider."""

