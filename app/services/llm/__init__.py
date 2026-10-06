"""LLM provider package exports and factory functions."""

from typing import Optional

from app.config import Settings, get_settings
from app.services.llm.base import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
)
from app.services.llm.huggingface import HuggingFaceProvider
from app.services.llm.ollama import OllamaProvider


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Instantiate and return the configured LLMProvider implementation."""
    cfg = settings or get_settings()
    if cfg.use_hf:
        return HuggingFaceProvider(
            token=cfg.hf_token,
            model=cfg.hf_model,
            base_url=cfg.hf_base_url,
            timeout=cfg.llm_timeout,
        )
    return OllamaProvider(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_model,
        timeout=cfg.llm_timeout,
    )


__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "LLMProviderError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    "LLMConfigurationError",
    "get_llm_provider",
]
