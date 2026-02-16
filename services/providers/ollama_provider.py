"""LLM provider implementation for Ollama."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx
from pydantic import ValidationError

from services.providers.base import LLMProvider
from services.providers.models import LLMResponseModel

LOGGER = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str,
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_seconds=backoff_seconds)
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=request_timeout,
        )

    async def summarize(self, transcript: str) -> LLMResponseModel:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous meeting assistant. Given a diarised transcript, "
                        "produce JSON with 'summary' (≤120 words) and 'actions' (each with "
                        "'task', 'assignee', 'due')."
                    ),
                },
                {"role": "user", "content": transcript}
            ],
            "stream": False
        }

        async def _request() -> LLMResponseModel:
            response = await self._client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            try:
                parsed = json.loads(content)
                return LLMResponseModel(
                    summary=parsed.get("summary", content),
                    actions=parsed.get("actions", []),
                )
            except json.JSONDecodeError:
                LOGGER.warning("Ollama returned non-JSON content, using raw text as summary")
                return LLMResponseModel(
                    summary=content,
                    actions=[],
                )

        return await self._run_with_retry(_request)

    async def close(self) -> None:
        await self._client.aclose()
