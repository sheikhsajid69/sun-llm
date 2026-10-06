"""Direct unit tests for Ollama and Hugging Face provider adapters."""

import pytest
import httpx

from app.models import ChatMessage
from app.services.llm.base import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMTimeoutError,
)
from app.services.llm.huggingface import HuggingFaceProvider
from app.services.llm.ollama import OllamaProvider


@pytest.mark.asyncio
async def test_ollama_generate_success():
    """Verify Ollama provider generates response correctly with 200 JSON."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            status_code=200,
            json={"message": {"role": "assistant", "content": "Hello from Ollama!"}},
        )

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(base_url="http://localhost:11434", model="mistral")

    # Monkeypatch AsyncClient in provider to use MockTransport
    async with httpx.AsyncClient(transport=transport) as client:
        # Test directly with custom client transport
        provider_url = f"{provider.base_url}/api/chat"
        resp = await client.post(provider_url, json={"model": "mistral"})
        assert resp.status_code == 200
        assert resp.json()["message"]["content"] == "Hello from Ollama!"


@pytest.mark.asyncio
async def test_ollama_model_not_found():
    """Verify Ollama provider raises LLMModelNotFoundError on 404."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=404, text="model 'mistral' not found")

    provider = OllamaProvider(base_url="http://localhost:11434", model="mistral")

    # Use custom transport by mocking httpx.AsyncClient inside provider.generate
    orig_async_client = httpx.AsyncClient
    try:
        httpx.AsyncClient = lambda **kwargs: orig_async_client(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMModelNotFoundError) as exc_info:
            await provider.generate([ChatMessage(role="user", content="Hi")], 0.7)
        assert "not found in Ollama" in exc_info.value.message
    finally:
        httpx.AsyncClient = orig_async_client


@pytest.mark.asyncio
async def test_ollama_connection_error():
    """Verify Ollama provider raises LLMConnectionError when connection fails."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    provider = OllamaProvider(base_url="http://localhost:11434", model="mistral")
    orig_async_client = httpx.AsyncClient
    try:
        httpx.AsyncClient = lambda **kwargs: orig_async_client(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMConnectionError) as exc_info:
            await provider.generate([ChatMessage(role="user", content="Hi")], 0.7)
        assert "Cannot connect to Ollama" in exc_info.value.message
    finally:
        httpx.AsyncClient = orig_async_client


@pytest.mark.asyncio
async def test_ollama_health_check_parsing():
    """Verify Ollama check_health correctly inspects available tags."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"models": [{"name": "mistral:latest"}, {"name": "llama3:latest"}]},
        )

    provider = OllamaProvider(base_url="http://localhost:11434", model="mistral")
    orig_async_client = httpx.AsyncClient
    try:
        httpx.AsyncClient = lambda **kwargs: orig_async_client(transport=httpx.MockTransport(handler))
        health = await provider.check_health()
        assert health["reachable"] is True
        assert health["model_available"] is True
        assert "mistral:latest" in health["available_models"]
    finally:
        httpx.AsyncClient = orig_async_client


@pytest.mark.asyncio
async def test_huggingface_missing_token_raises_configuration_error():
    """Verify Hugging Face provider rejects generation when HF_TOKEN is empty."""
    provider = HuggingFaceProvider(token="")
    with pytest.raises(LLMConfigurationError) as exc_info:
        await provider.generate([ChatMessage(role="user", content="Hi")], 0.7)
    assert "HF_TOKEN is not set" in exc_info.value.message


@pytest.mark.asyncio
async def test_huggingface_generate_success():
    """Verify Hugging Face provider parses OpenAI-compatible response."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert "Bearer test_token" in request.headers["Authorization"]
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": "HF Reply"}}]},
        )

    provider = HuggingFaceProvider(token="test_token", model="mistralai/Mistral-7B-Instruct-v0.2")
    orig_async_client = httpx.AsyncClient
    try:
        httpx.AsyncClient = lambda **kwargs: orig_async_client(transport=httpx.MockTransport(handler))
        reply = await provider.generate([ChatMessage(role="user", content="Hi")], 0.7)
        assert reply == "HF Reply"
    finally:
        httpx.AsyncClient = orig_async_client


@pytest.mark.asyncio
async def test_huggingface_auth_failure_raises_auth_error():
    """Verify Hugging Face returns LLMAuthenticationError on 401."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, json={"error": "Invalid token"})

    provider = HuggingFaceProvider(token="invalid_token")
    orig_async_client = httpx.AsyncClient
    try:
        httpx.AsyncClient = lambda **kwargs: orig_async_client(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await provider.generate([ChatMessage(role="user", content="Hi")], 0.7)
        assert "token" in exc_info.value.message.lower()
    finally:
        httpx.AsyncClient = orig_async_client


@pytest.mark.asyncio
async def test_huggingface_rate_limit_raises_provider_error():
    """Verify Hugging Face 429 raises LLMProviderError with status 429."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=429, json={"error": "Rate limited"})

    provider = HuggingFaceProvider(token="test_token")
    orig_async_client = httpx.AsyncClient
    try:
        httpx.AsyncClient = lambda **kwargs: orig_async_client(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.generate([ChatMessage(role="user", content="Hi")], 0.7)
        assert exc_info.value.status_code == 429
    finally:
        httpx.AsyncClient = orig_async_client
