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

    def __init__(
        self,
        token: str = "",
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        base_url: str = "https://api-inference.huggingface.co/models",
        timeout: float = 120.0,
    ):
        self.token = token.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self.model

    def _get_headers(self) -> dict[str, str]:
        if not self.token:
            raise LLMConfigurationError(
                "Hugging Face inference is enabled but HF_TOKEN is not set. "
                "Provide a valid token in HF_TOKEN environment variable.",
                provider="huggingface",
            )
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        """Call Hugging Face Chat Completions endpoint."""
        headers = self._get_headers()

        # Resolve OpenAI-compatible endpoint on Hugging Face inference API
        if self.base_url.endswith("/v1/chat/completions") or self.base_url.endswith("/chat/completions"):
            url = self.base_url
        else:
            url = f"{self.base_url}/{self.model}/v1/chat/completions"

        # Safe temperature clamping: some HF models require temperature > 0.0 or <= 1.5
        clamped_temp = max(0.01, min(temperature, 1.99)) if temperature > 0 else 0.01

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": 1024,
            "temperature": clamped_temp,
        }

        logger.debug("Dispatching request to Hugging Face endpoint: %s", url)

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

        if response.status_code in (401, 403):
            raise LLMAuthenticationError(
                "Invalid or expired Hugging Face token. Please verify HF_TOKEN.",
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
        """Check Hugging Face configuration and model connectivity."""
        if not self.token:
            return {
                "reachable": False,
                "configured": False,
                "error": "HF_TOKEN is not configured in environment.",
            }

        # Check model reachability via Hugging Face Hub API
        meta_url = f"https://huggingface.co/api/models/{self.model}"
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
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
