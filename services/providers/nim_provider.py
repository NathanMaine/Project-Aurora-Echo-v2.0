"""NVIDIA NIM provider for LLM inference."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import ValidationError

from .base import LLMProvider
from .models import LLMResponseModel

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meticulous meeting assistant. Given a diarised transcript, "
    "produce JSON with 'summary' (≤120 words) and 'actions' (each with "
    "'task', 'assignee', 'due')."
)


class NIMProvider(LLMProvider):
    """Provider for NVIDIA NIM inference microservices."""

    name = "nim"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/chat/completions",
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_seconds=backoff_seconds)
        self._base_url = base_url.rstrip('/')
        self._model = model
        self._api_key = api_key
        self._endpoint = endpoint.lstrip('/')
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def summarize(self, transcript: str) -> LLMResponseModel:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.1,
            "max_tokens": 600,
        }

        async def _request() -> LLMResponseModel:
            response = await self._client.post(
                f"/{self._endpoint}",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError("No choices in NIM response")

            content = data["choices"][0]["message"]["content"]
            payload_dict = json.loads(content)
            return LLMResponseModel.parse_obj(payload_dict)

        return await self._run_with_retry(_request)

    async def close(self) -> None:
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"NIMProvider(base_url={self._base_url}, model={self._model})"
