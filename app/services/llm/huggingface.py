"""Hugging Face Inference Provider implementation.

Supports Hugging Face serverless / dedicated chat completions inference.
Adheres to the OpenAI-compatible v1 chat completions endpoint standard.
"""

import logging
from typing import Any

import httpx

from app.models import ChatMessage
from app.services.llm.base import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class HuggingFaceProvider(LLMProvider):
    """Provider for Hugging Face Inference API."""

    DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
    FALLBACK_MODELS = (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "zai-org/GLM-5.3-Flash",
        "Qwen/Qwen3.5-9B",
        "microsoft/phi-4",
        "meta-llama/Llama-3.1-8B-Instruct",
    )

    def __init__(
        self,
        token: str = "",
        model: str = DEFAULT_MODEL,
        base_url: str = "https://router.huggingface.co/v1/chat/completions",
        timeout: float = 120.0,
    ):
        self.token = token.strip()
        self.model = model.strip() or self.DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "groq" if self.is_groq else "huggingface"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def is_groq(self) -> bool:
        """Detect whether the current provider configuration targets Groq."""
        return (
            self.token.startswith("gsk_")
            or "groq.com" in self.base_url
            or self.model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768")
        )

    def _get_headers(self) -> dict[str, str]:
        if not self.token:
            name = "Groq" if self.is_groq else "Hugging Face"
            raise LLMConfigurationError(
                f"{name} inference is enabled but HF_TOKEN is not set. "
                "Provide a valid token in HF_TOKEN environment variable.",
                provider="groq" if self.is_groq else "huggingface",
            )
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _resolve_url(self) -> str:
        """Resolve modern OpenAI-compatible router/endpoint."""
        url = self.base_url.strip()

        # Handle Groq API endpoint
        if self.is_groq:
            if "groq.com" in url:
                if url.endswith("/chat/completions"):
                    return url
                if url.endswith("/v1"):
                    return f"{url}/chat/completions"
                return f"{url}/openai/v1/chat/completions"
            return "https://api.groq.com/openai/v1/chat/completions"

        # Handle Hugging Face Router endpoint
        if "api-inference.huggingface.co" in url or "hf-inference" in url or not url:
            return "https://router.huggingface.co/v1/chat/completions"
        if url.endswith("/v1/chat/completions") or url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        """Call Chat Completions endpoint with automated multi-model fallback."""
        headers = self._get_headers()
        url = self._resolve_url()

        # Safe temperature clamping
        clamped_temp = max(0.01, min(temperature, 1.99)) if temperature > 0 else 0.01

        # Build sequence of models to try
        if self.is_groq:
            groq_fallbacks = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768")
            models_to_try = [self.model]
            for m in groq_fallbacks:
                if m not in models_to_try:
                    models_to_try.append(m)
        else:
            models_to_try = [self.model]
            for m in self.FALLBACK_MODELS:
                if m not in models_to_try:
                    models_to_try.append(m)

        last_response = None
        for attempt_model in models_to_try:
            payload = {
                "model": attempt_model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": 1024,
                "temperature": clamped_temp,
            }

            logger.debug("Dispatching request to Hugging Face endpoint: %s with model %s", url, attempt_model)

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
            except httpx.ConnectError as err:
                logger.error("Failed to connect to Hugging Face API: %s", err)
                raise LLMConnectionError(
                    "Cannot connect to Hugging Face Inference API. Check network connectivity.",
                    provider="huggingface",
                ) from err
            except httpx.TimeoutException as err:
                logger.error("Hugging Face generation timed out after %s seconds", self.timeout)
                raise LLMTimeoutError(
                    f"Hugging Face request timed out after {self.timeout}s.",
                    provider="huggingface",
                ) from err
            except Exception as err:
                logger.error("Unexpected error contacting Hugging Face: %s", err)
                raise LLMProviderError(
                    f"Unexpected error communicating with Hugging Face: {str(err)}",
                    provider="huggingface",
                ) from err

            last_response = response

            # Check if this model is unsupported by providers on the router
            is_unsupported = (
                response.status_code == 400
                and any(k in response.text.lower() for k in ("not supported", "model", "provider"))
            ) or response.status_code in (404, 422)

            if is_unsupported:
                logger.warning(
                    "Model '%s' returned status %d ('%s'). Retrying with fallback model...",
                    attempt_model,
                    response.status_code,
                    response.text[:120],
                )
                continue

            # Model is supported and response received; update active model and exit attempt loop
            self.model = attempt_model
            break

        response = last_response
        if response is None:
            raise LLMProviderError("No response received from Hugging Face.", provider="huggingface")

        if response.status_code in (401, 403):
            raise LLMAuthenticationError(
                "Invalid or expired Hugging Face token. Please verify HF_TOKEN.",
                provider="huggingface",
            )
        elif response.status_code == 402:
            raise LLMProviderError(
                "Hugging Face account has exhausted its Inference Provider credits. "
                "Add pre-paid credits at https://huggingface.co/settings/billing or use a free provider like Groq.",
                status_code=402,
                provider="huggingface",
            )
        elif response.status_code == 404:
            raise LLMModelNotFoundError(
                f"Hugging Face model '{self.model}' not found or chat interface not supported.",
                provider="huggingface",
            )
        elif response.status_code == 429:
            raise LLMProviderError(
                "Hugging Face rate limit reached. Please wait and try again.",
                status_code=429,
                provider="huggingface",
            )
        elif response.status_code == 503:
            # Model loading state often returns estimated_time in JSON
            try:
                err_data = response.json()
                est = err_data.get("estimated_time", "briefly")
                msg = f"Model '{self.model}' is currently loading on Hugging Face (estimated: {est}s). Please retry."
            except Exception:
                msg = f"Hugging Face service unavailable or model loading: {response.text[:150]}"
            raise LLMProviderError(msg, status_code=503, provider="huggingface")
        elif response.status_code != 200:
            raise LLMProviderError(
                f"Hugging Face returned status {response.status_code}: {response.text[:200]}",
                status_code=502,
                provider="huggingface",
            )

        try:
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMProviderError(
                    "Empty choices array returned from Hugging Face.",
                    provider="huggingface",
                )
            reply = (choices[0].get("message") or {}).get("content", "").strip()
            if not reply:
                raise LLMProviderError(
                    "Hugging Face returned an empty message content.",
                    provider="huggingface",
                )
            return reply
        except LLMProviderError:
            raise
        except Exception as err:
            raise LLMProviderError(
                f"Failed to parse Hugging Face response: {str(err)}",
                provider="huggingface",
            ) from err

    async def check_health(self) -> dict[str, Any]:
        """Check provider configuration and model connectivity."""
        if not self.token:
            provider_label = "Groq" if self.is_groq else "HF_TOKEN"
            return {
                "reachable": False,
                "configured": False,
                "error": f"{provider_label} token is not configured in environment.",
            }

        headers = self._get_headers()

        # Branch for Groq API
        if self.is_groq:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
                if resp.status_code == 200:
                    return {
                        "reachable": True,
                        "configured": True,
                        "provider": "groq",
                        "model": self.model,
                    }
                elif resp.status_code in (401, 403):
                    return {
                        "reachable": False,
                        "configured": True,
                        "error": "Authentication token was rejected by Groq. Please check your Groq API key.",
                    }
                return {
                    "reachable": False,
                    "configured": True,
                    "error": f"Groq API returned HTTP {resp.status_code}",
                }
            except Exception as err:
                return {
                    "reachable": False,
                    "configured": True,
                    "error": f"Cannot connect to Groq: {err}",
                }

        # Branch for Hugging Face Hub API
        meta_url = f"https://huggingface.co/api/models/{self.model}"
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(meta_url, headers=headers)
            if resp.status_code == 200:
                return {
                    "reachable": True,
                    "configured": True,
                    "model": self.model,
                }
            elif resp.status_code in (401, 403):
                return {
                    "reachable": False,
                    "configured": True,
                    "error": "Authentication token was rejected by Hugging Face.",
                }
            elif resp.status_code == 404:
                return {
                    "reachable": False,
                    "configured": True,
                    "error": f"Model '{self.model}' does not exist on Hugging Face.",
                }
            return {
                "reachable": False,
                "configured": True,
                "error": f"Hugging Face API returned HTTP {resp.status_code}",
            }
        except Exception as err:
            return {
                "reachable": False,
                "configured": True,
                "error": str(err),
            }
