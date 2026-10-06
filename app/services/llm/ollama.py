"""Ollama LLM Provider implementation.

Interfaces with a local or network Ollama daemon via the /api/chat HTTP endpoint.
No external API keys required.
"""

import logging
from typing import Any

import httpx

from app.models import ChatMessage
from app.services.llm.base import (
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Provider for local/remote Ollama inference instances."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        """Call Ollama /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }

        logger.debug("Dispatching request to Ollama endpoint: %s with model %s", url, self.model)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as err:
            logger.error("Failed to connect to Ollama at %s: %s", self.base_url, err)
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure the Ollama daemon is running.",
                provider="ollama",
            ) from err
        except httpx.TimeoutException as err:
            logger.error("Ollama generation timed out after %s seconds", self.timeout)
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.timeout}s.",
                provider="ollama",
            ) from err
        except Exception as err:
            logger.error("Unexpected network error while contacting Ollama: %s", err)
            raise LLMProviderError(
                f"Unexpected error communicating with Ollama: {str(err)}",
                provider="ollama",
            ) from err

        if response.status_code == 404:
            raise LLMModelNotFoundError(
                f"Model '{self.model}' was not found in Ollama. Pull it first via: 'ollama pull {self.model}'",
                provider="ollama",
            )
        elif response.status_code != 200:
            error_text = response.text[:200]
            raise LLMProviderError(
                f"Ollama returned HTTP {response.status_code}: {error_text}",
                status_code=502,
                provider="ollama",
            )

        try:
            data = response.json()
        except Exception as err:
            raise LLMProviderError(
                "Ollama returned invalid JSON response.",
                provider="ollama",
            ) from err

        if "error" in data:
            raise LLMProviderError(
                f"Ollama reported error: {data['error']}",
                status_code=502,
                provider="ollama",
            )

        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise LLMProviderError(
                "Ollama returned an empty response.",
                provider="ollama",
            )

        return content

    async def check_health(self) -> dict[str, Any]:
        """Verify reachability and model availability via /api/tags."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                tags = [m.get("name") for m in data.get("models", [])]
                model_found = any(self.model in tag for tag in tags)
                return {
                    "reachable": True,
                    "model_available": model_found,
                    "available_models": tags,
                    "configured_model": self.model,
                }
            return {
                "reachable": False,
                "status_code": resp.status_code,
                "error": f"Ollama returned HTTP {resp.status_code}",
            }
        except httpx.ConnectError:
            return {
                "reachable": False,
                "error": f"Connection refused at {self.base_url}",
            }
        except httpx.TimeoutException:
            return {
                "reachable": False,
                "error": "Connection timed out checking Ollama status",
            }
        except Exception as err:
            return {
                "reachable": False,
                "error": str(err),
            }
