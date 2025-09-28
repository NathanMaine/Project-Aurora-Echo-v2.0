"""LLM provider implementation for Azure OpenAI."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from pydantic import ValidationError

from services.providers.base import LLMProvider
from services.providers.models import LLMResponseModel

LOGGER = logging.getLogger(__name__)


class AzureOpenAIProvider(LLMProvider):
    name = "azure-openai"

    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        request_timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_seconds=backoff_seconds)
        base_url = endpoint.rstrip("/")
        path = f"/openai/deployments/{deployment}/chat/completions"
        self._url = f"{base_url}{path}?api-version={api_version}"
        self._client = httpx.AsyncClient(timeout=request_timeout)
        self._api_key = api_key

    async def summarize(self, transcript: str) -> LLMResponseModel:
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous meeting assistant. Provide JSON with a 'summary' "
                        "(≤120 words) and 'actions' array (each item has 'task', 'assignee', 'due')."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.1,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }

        async def _request() -> LLMResponseModel:
            response = await self._client.post(self._url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            payload_dict = json.loads(content)
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
                    LOGGER.exception("Azure OpenAI provider failed after %s attempts", attempt)
                    raise
                LOGGER.warning(
                    "Azure OpenAI request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def close(self) -> None:
        await self._client.aclose()
