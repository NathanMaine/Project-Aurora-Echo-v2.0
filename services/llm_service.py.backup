"""Multi-provider LLM service with retry/backoff and pydantic validation."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from services.providers.base import LLMProvider
from services.providers.models import LLMResponseModel
from services.providers.vllm_provider import VLLMProvider
from services.providers.ollama_provider import OllamaProvider
from services.providers.xai_grok_provider import XAIGrokProvider
from services.providers.openai_provider import OpenAIProvider
from services.providers.azure_openai_provider import AzureOpenAIProvider
from services.providers.anthropic_provider import AnthropicClaudeProvider
from services.providers.gemini_provider import GeminiProvider

LOGGER = logging.getLogger(__name__)


class LLMService:
    """Coordinates multiple LLM providers with failover and retries."""

    def __init__(
        self,
        *,
        provider_order: Optional[str] = None,
        max_retries: Optional[int] = None,
        backoff_seconds: Optional[float] = None,
    ) -> None:
        order_env = provider_order or os.getenv("LLM_PROVIDER_ORDER", "vllm,grok")
        self._provider_names = [name.strip().lower() for name in order_env.split(",") if name.strip()]
        if not self._provider_names:
            self._provider_names = ["vllm", "grok"]

        self._max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "3"))
        self._backoff_seconds = backoff_seconds or float(os.getenv("LLM_BACKOFF_SECONDS", "1.0"))

        self._vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001")
        self._vllm_model_id = os.getenv("VLLM_MODEL_ID", "meta-llama-3-8b-instruct")
        self._vllm_endpoint = os.getenv("VLLM_COMPLETIONS_ENDPOINT", "/v1/chat/completions")
        self._vllm_api_key = os.getenv("VLLM_API_KEY")

        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_model_id = os.getenv("OLLAMA_MODEL_ID", "llama3.1:8b")

        self._xai_api_key = os.getenv("XAI_API_KEY")

        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._openai_model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
        self._openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")

        self._azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self._azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self._anthropic_model_id = os.getenv("ANTHROPIC_MODEL_ID", "claude-3-sonnet-20240229")

        self._gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        self._gemini_model_id = os.getenv("GOOGLE_GEMINI_MODEL_ID", "gemini-1.5-pro-latest")

        self._providers: List[LLMProvider] = []
        self._initialise_providers()

    def _initialise_providers(self) -> None:
        for name in self._provider_names:
            if name == "vllm":
                provider = VLLMProvider(
                    base_url=self._vllm_base_url,
                    model=self._vllm_model_id,
                    api_key=self._vllm_api_key,
                    endpoint=self._vllm_endpoint,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            elif name == "ollama":
                provider = OllamaProvider(
                    base_url=self._ollama_base_url,
                    model=self._ollama_model_id,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            elif name == "grok":
                if not self._xai_api_key:
                    LOGGER.warning("Skipping Grok provider because XAI_API_KEY is not set")
                    continue
                provider = XAIGrokProvider(
                    api_key=self._xai_api_key,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            elif name in {"openai", "openai-o1"}:
                if not self._openai_api_key:
                    LOGGER.warning("Skipping OpenAI provider because OPENAI_API_KEY is not set")
                    continue
                provider = OpenAIProvider(
                    api_key=self._openai_api_key,
                    model=self._openai_model_id,
                    base_url=self._openai_base_url,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            elif name in {"azure", "azure-openai"}:
                if not (self._azure_openai_endpoint and self._azure_openai_deployment and self._azure_openai_api_key):
                    LOGGER.warning(
                        "Skipping Azure OpenAI provider because AZURE_OPENAI_ENDPOINT/DEPLOYMENT/API_KEY are not fully set"
                    )
                    continue
                provider = AzureOpenAIProvider(
                    endpoint=self._azure_openai_endpoint,
                    deployment=self._azure_openai_deployment,
                    api_key=self._azure_openai_api_key,
                    api_version=self._azure_openai_api_version,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            elif name in {"claude", "anthropic"}:
                if not self._anthropic_api_key:
                    LOGGER.warning("Skipping Claude provider because ANTHROPIC_API_KEY is not set")
                    continue
                provider = AnthropicClaudeProvider(
                    api_key=self._anthropic_api_key,
                    model=self._anthropic_model_id,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            elif name == "gemini":
                if not self._gemini_api_key:
                    LOGGER.warning("Skipping Gemini provider because GOOGLE_GEMINI_API_KEY is not set")
                    continue
                provider = GeminiProvider(
                    api_key=self._gemini_api_key,
                    model=self._gemini_model_id,
                    max_retries=self._max_retries,
                    backoff_seconds=self._backoff_seconds,
                )
                self._providers.append(provider)
            else:
                LOGGER.warning("Unknown LLM provider '%s' in LLM_PROVIDER_ORDER; skipping", name)

        if not self._providers:
            LOGGER.error("No LLM providers configured; summaries will be unavailable")

    async def summarize_meeting(self, transcript: str) -> Optional[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for provider in self._providers:
            try:
                LOGGER.info("Invoking LLM provider '%s'", provider.name)
                response: LLMResponseModel = await provider.summarize(transcript)
                return response.dict()
            except Exception as exc:  # pragma: no cover - runtime logging only
                last_error = exc
                LOGGER.exception("Provider '%s' failed: %s", provider.name, exc)
                continue
        if last_error:
            LOGGER.error("All LLM providers failed; returning None")
        return None

    async def close(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception as exc:  # pragma: no cover - runtime logging only
                LOGGER.warning("Failed to close provider '%s': %s", provider.name, exc)
