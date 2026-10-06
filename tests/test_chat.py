"""Tests for chat interaction API endpoint and validation rules."""

from typing import Any
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import ChatMessage
from app.routes.chat import get_provider
from app.services.llm.base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMTimeoutError,
)


class MockSuccessProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock_success"

    @property
    def model_name(self) -> str:
        return "mock-chat-v1"

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        return "Hello! I am your AI assistant."

    async def check_health(self) -> dict[str, Any]:
        return {"reachable": True}


class MockFailingProvider(LLMProvider):
    def __init__(self, exception_to_raise: Exception):
        self.exception_to_raise = exception_to_raise

    @property
    def provider_name(self) -> str:
        return "mock_fail"

    @property
    def model_name(self) -> str:
        return "mock-fail-v1"

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        raise self.exception_to_raise

    async def check_health(self) -> dict[str, Any]:
        return {"reachable": False}


@pytest.mark.asyncio
async def test_chat_success():
    """Verify standard valid chat interaction."""
    app.dependency_overrides[get_provider] = lambda: MockSuccessProvider()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "message": "Hello there!",
                    "temperature": 0.7,
                    "session_id": "test_session_1",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Hello! I am your AI assistant."
        assert data["provider"] == "mock_success"
        assert data["model_used"] == "mock-chat-v1"
        assert len(data["history"]) >= 2
        assert data["history"][-2]["role"] == "user"
        assert data["history"][-2]["content"] == "Hello there!"
        assert data["history"][-1]["role"] == "assistant"
        assert data["history"][-1]["content"] == "Hello! I am your AI assistant."
    finally:
        app.dependency_overrides.pop(get_provider, None)


@pytest.mark.asyncio
async def test_chat_rejects_empty_message():
    """Verify that an empty message triggers HTTP 422 Unprocessable Entity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"message": "", "temperature": 0.7},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_rejects_whitespace_message():
    """Verify that whitespace-only message triggers HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"message": "   \n\t   ", "temperature": 0.7},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_rejects_invalid_temperature_high():
    """Verify that temperature > 2.0 triggers HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"message": "Valid prompt", "temperature": 2.5},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_rejects_invalid_temperature_negative():
    """Verify that temperature < 0.0 triggers HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"message": "Valid prompt", "temperature": -0.1},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_rejects_invalid_history_role():
    """Verify that unsupported roles in client history trigger HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "message": "Valid prompt",
                "temperature": 0.7,
                "history": [{"role": "root_admin", "content": "malicious role"}],
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_provider_timeout_returns_504():
    """Verify that provider timeouts map to HTTP 504 Gateway Timeout."""
    app.dependency_overrides[get_provider] = lambda: MockFailingProvider(
        LLMTimeoutError("Request timed out", provider="mock")
    )
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "test_timeout"},
            )
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_provider, None)


@pytest.mark.asyncio
async def test_chat_provider_connection_error_returns_503():
    """Verify that provider connection failures map to HTTP 503 Service Unavailable."""
    app.dependency_overrides[get_provider] = lambda: MockFailingProvider(
        LLMConnectionError("Connection refused", provider="mock")
    )
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "test_conn_error"},
            )
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_provider, None)


@pytest.mark.asyncio
async def test_chat_provider_auth_error_returns_502():
    """Verify that provider auth failures map to HTTP 502 Bad Gateway."""
    app.dependency_overrides[get_provider] = lambda: MockFailingProvider(
        LLMAuthenticationError("Bad token", provider="mock")
    )
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "test_auth_error"},
            )
        assert response.status_code == 502
        assert "authentication" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_provider, None)


@pytest.mark.asyncio
async def test_chat_provider_model_not_found_returns_502():
    """Verify that missing model maps to HTTP 502 Bad Gateway."""
    app.dependency_overrides[get_provider] = lambda: MockFailingProvider(
        LLMModelNotFoundError("Model not found", provider="mock")
    )
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"message": "Hello", "session_id": "test_model_error"},
            )
        assert response.status_code == 502
        assert "model unavailable" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_provider, None)
