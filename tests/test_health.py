"""Tests for root UI and health diagnostic endpoints."""

from typing import Any
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import ChatMessage
from app.routes.chat import get_provider
from app.services.llm.base import LLMProvider


class MockHealthyProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock_provider"

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        return "Mock response"

    async def check_health(self) -> dict[str, Any]:
        return {"reachable": True, "model_available": True}


class MockDegradedProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock_provider"

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    async def generate(self, messages: list[ChatMessage], temperature: float) -> str:
        return "Mock response"

    async def check_health(self) -> dict[str, Any]:
        return {"reachable": False, "error": "Connection refused"}


@pytest.mark.asyncio
async def test_root_endpoint_returns_html():
    """Verify that GET / serves the chat UI HTML page."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "LLM Chatbot" in response.text


@pytest.mark.asyncio
async def test_health_endpoint_healthy_provider():
    """Verify that GET /health returns status 'ok' when provider is reachable."""
    app.dependency_overrides[get_provider] = lambda: MockHealthyProvider()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["provider"] == "mock_provider"
        assert data["model"] == "mock-model-v1"
        assert data["details"]["reachable"] is True
    finally:
        app.dependency_overrides.pop(get_provider, None)


@pytest.mark.asyncio
async def test_health_endpoint_degraded_provider():
    """Verify that GET /health returns status 'degraded' when provider is offline."""
    app.dependency_overrides[get_provider] = lambda: MockDegradedProvider()
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["provider"] == "mock_provider"
        assert data["details"]["reachable"] is False
    finally:
        app.dependency_overrides.pop(get_provider, None)
