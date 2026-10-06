"""Abstract Base Provider for Large Language Model integrations."""

from abc import ABC, abstractmethod
import logging
from typing import Any

from app.models import ChatMessage

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""

    def __init__(self, message: str, status_code: int = 502, provider: str = "llm"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider


class LLMConnectionError(LLMProviderError):
    """Raised when the LLM provider host or service is unreachable."""

    def __init__(self, message: str, provider: str = "llm"):
        super().__init__(message, status_code=503, provider=provider)


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""

    def __init__(self, message: str, provider: str = "llm"):
        super().__init__(message, status_code=504, provider=provider)


class LLMAuthenticationError(LLMProviderError):
    """Raised when an LLM provider rejects authentication credentials."""

    def __init__(self, message: str, provider: str = "llm"):
        super().__init__(message, status_code=502, provider=provider)


class LLMModelNotFoundError(LLMProviderError):
    """Raised when the requested model is not found on the provider."""

    def __init__(self, message: str, provider: str = "llm"):
        super().__init__(message, status_code=502, provider=provider)


class LLMConfigurationError(LLMProviderError):
    """Raised when provider configuration or secrets are missing/invalid."""

    def __init__(self, message: str, provider: str = "llm"):
        super().__init__(message, status_code=500, provider=provider)


class LLMProvider(ABC):
    """Abstract interface defining the contract for LLM backends."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider backend."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier used by this provider."""
        ...

    @abstractmethod
    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        """Generate a response given a conversation history and temperature.

        Args:
            messages: Formatted list of ChatMessage instances.
            temperature: Sampling temperature (0.0 - 2.0).

        Returns:
            Assistant response string.

        Raises:
            LLMProviderError or subclasses on failure.
        """
        ...

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Perform diagnostic check on the provider availability.

        Returns:
            Dictionary with health status information.
        """
        ...
