"""Pydantic schemas and validation models for LLM Prompt Project."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Permitted conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Individual conversational message turn."""

    role: str = Field(
        ...,
        description="Message author role: user, assistant, or system",
        examples=["user", "assistant"]
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text content of the message",
        examples=["Hello, how can you help me?"]
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {r.value for r in MessageRole}
        if normalized not in allowed:
            raise ValueError(f"Invalid message role '{value}'. Must be one of {sorted(allowed)}")
        return normalized

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message content cannot be empty or whitespace only.")
        return cleaned


class ChatRequest(BaseModel):
    """Chat interaction request payload."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Prompt or query sent by the user",
        examples=["Explain prompt engineering in simple terms."]
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0.0 (deterministic) and 2.0 (creative)"
    )
    history: Optional[list[ChatMessage]] = Field(
        default=None,
        description="Optional client-side conversation history for synchronization"
    )
    session_id: str = Field(
        default="default",
        max_length=128,
        description="Identifier for multi-turn conversational session"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or contain only whitespace.")
        return cleaned


class ChatResponse(BaseModel):
    """Successful chat response payload."""

    reply: str = Field(..., description="Assistant generated response text")
    model_used: str = Field(..., description="LLM model identifier used for generation")
    provider: str = Field(..., description="LLM provider name (ollama or huggingface)")
    history: list[ChatMessage] = Field(
        ..., description="Updated bounded conversation history"
    )


class HealthResponse(BaseModel):
    """Application health and provider diagnostic status."""

    status: str = Field(..., description="Overall health status: ok, degraded, or error")
    provider: str = Field(..., description="Configured LLM provider name")
    model: str = Field(..., description="Configured LLM model identifier")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Diagnostic details or connectivity notes"
    )

