"""LLM provider implementation for Google Gemini."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from pydantic import ValidationError

from services.providers.base import LLMProvider
from services.providers.models import LLMResponseModel

LOGGER = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        request_timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_seconds=backoff_seconds)
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url, timeout=request_timeout)
        self._model = model

    async def summarize(self, transcript: str) -> LLMResponseModel:
        endpoint = f"/v1beta/models/{self._model}:generateContent"
        params = {"key": self._api_key}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "You are a meticulous meeting assistant. Return JSON with 'summary' (≤120 words) "
                                "and an 'actions' array (each action has 'task', 'assignee', 'due').\n\n"
                                f"Transcript: {transcript}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 600,
                "responseMimeType": "application/json",
            },
        }

        async def _request() -> LLMResponseModel:
            response = await self._client.post(endpoint, params=params, json=payload)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned no candidates")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            combined_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            payload_dict = json.loads(combined_text)
            return LLMResponseModel.parse_obj(payload_dict)

        return await self._run_with_retry(_request)

    async def _run_with_retry(self, func):
        delay = self.backoff_seconds
        attempt = 0
        while True:
            try:
                return await func()
            except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError) as exc:
                attempt += 1
                if attempt >= self.max_retries:
                    LOGGER.exception("Gemini provider failed after %s attempts", attempt)
                    raise
                LOGGER.warning(
                    "Gemini request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def close(self) -> None:
        await self._client.aclose()
