"""LLM provider implementation for local vLLM deployments."""

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


class VLLMProvider(LLMProvider):
    name = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/chat/completions",
        request_timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_seconds=backoff_seconds)
        self._model = model
        self._api_key = api_key
        self._endpoint = endpoint
        self._client = httpx.AsyncClient(base_url=base_url, timeout=request_timeout)

    async def summarize(self, transcript: str) -> LLMResponseModel:
        headers = {}
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
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.1,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }

        async def _request() -> LLMResponseModel:
            response = await self._client.post(self._endpoint, json=payload, headers=headers or None)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                # vLLM may return content as a list of message parts
                content = "".join(part.get("text", "") for part in content)
            try:
                payload_dict = json.loads(content)
            except json.JSONDecodeError as exc:  # pragma: no cover - runtime logging only
                LOGGER.error("vLLM returned non-JSON content: %s", content)
                raise
            return LLMResponseModel.parse_obj(payload_dict)

        return await self._run_with_retry(_request)

    async def _run_with_retry(self, func):
        delay = self.backoff_seconds
        attempt = 0
        while True:
            try:
                return await func()
            except (httpx.HTTPError, json.JSONDecodeError, ValidationError) as exc:
                attempt += 1
                if attempt >= self.max_retries:
                    LOGGER.exception("vLLM provider failed after %s attempts", attempt)
                    raise
                LOGGER.warning(
                    "vLLM request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def close(self) -> None:
        await self._client.aclose()

