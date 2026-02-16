"""NVIDIA NIM provider for LLM inference."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import LLMProvider
from .models import LLMResponseModel

LOGGER = logging.getLogger(__name__)


class NIMProvider(LLMProvider):
    """Provider for NVIDIA NIM inference microservices."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        endpoint: str = "/v1/chat/completions",
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
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make request to NIM endpoint."""
        url = f"{self.base_url}/{self.endpoint}"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Add any extra parameters
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value

        LOGGER.debug("Sending request to NIM at %s", url)
        response = self.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponseModel:
        """Generate a response using NIM."""
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                data = self._make_request(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                latency = time.time() - start_time

                # Extract response text
                if "choices" not in data or len(data["choices"]) == 0:
                    raise ValueError("No choices in NIM response")
                choice = data["choices"][0]
                if "message" not in choice:
                    raise ValueError("No message in choice")
                message = choice["message"]
                content = message.get("content", "")

                # Extract usage
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                return LLMResponseModel(
                    content=content,
                    latency=latency,
                    provider="nim",
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw_response=data,
                )
            except Exception as e:
                LOGGER.warning(
                    "Attempt %d/%d failed for NIM provider: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_seconds * (2 ** attempt))
        raise RuntimeError("Should not reach here")

    def __repr__(self) -> str:
        return f"NIMProvider(base_url={self.base_url}, model={self.model})"
