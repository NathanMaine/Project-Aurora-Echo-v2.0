"""LLM provider implementation for Ollama."""

from __future__ import annotations

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
        self._timeout = request_timeout

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
                        "provide a structured summary with key points, decisions made, and "
                        "action items with assignees and due dates. Format your response as JSON with "
                        "the following structure: {\"summary\": \"string\", \"action_items\": [{\"task\": \"string\", \"assignee\": \"string\", \"due_date\": \"string\"}]}"
                    ),
                },
                {"role": "user", "content": transcript}
            ],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Try to parse as JSON, fallback to plain text
            try:
                parsed = json.loads(content)
                return LLMResponseModel(
                    content=parsed.get("summary", content),
                    action_items=parsed.get("action_items", [])
                )
            except json.JSONDecodeError:
                return LLMResponseModel(
                    content=content,
                    action_items=[]
                )

    async def close(self) -> None:
        pass  # httpx client is managed with context manager