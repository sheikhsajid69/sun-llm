"""Chat and health API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.conversation import ConversationService
from app.services.llm import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    get_llm_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Singleton conversation service container
_conversation_service_instance: ConversationService | None = None


def get_conversation_service(
    settings: Annotated[Settings, Depends(get_settings)]
) -> ConversationService:
    """Dependency provider for ConversationService."""
    global _conversation_service_instance
    if _conversation_service_instance is None:
        _conversation_service_instance = ConversationService(settings)
    return _conversation_service_instance


def get_provider(
    settings: Annotated[Settings, Depends(get_settings)]
) -> LLMProvider:
    """Dependency provider for LLMProvider."""
    return get_llm_provider(settings)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    provider: Annotated[LLMProvider, Depends(get_provider)]
) -> HealthResponse:
    """Health diagnostic endpoint.

    Distinguishes application operational health from backend provider reachability.
    Does not crash or return 500 when optional external providers are offline.
    """
    logger.debug("Executing health check for provider: %s", provider.provider_name)
    try:
        diagnostics = await provider.check_health()
        is_reachable = diagnostics.get("reachable", False)
        health_status = "ok" if is_reachable else "degraded"
        return HealthResponse(
            status=health_status,
            provider=provider.provider_name,
            model=provider.model_name,
            details=diagnostics,
        )
    except Exception as err:
        logger.warning("Error during provider health evaluation: %s", err)
        return HealthResponse(
            status="degraded",
            provider=provider.provider_name,
            model=provider.model_name,
            details={"reachable": False, "error": str(err)},
        )


@router.post("/api/chat", response_model=ChatResponse)
async def chat_interaction(
    request: ChatRequest,
    conv_service: Annotated[ConversationService, Depends(get_conversation_service)],
    provider: Annotated[LLMProvider, Depends(get_provider)],
) -> ChatResponse:
    """Primary chat endpoint.

    Validates request payload, synchronizes optional frontend history,
    assembles prompts within memory boundaries, invokes the LLM provider,
    and returns assistant reply with updated history.
    """
    # 1. Synchronize client history if submitted
    if request.history is not None:
        conv_service.sync_client_history(request.session_id, request.history)

    # 2. Build complete message payload (system prompt + history + user message)
    messages = conv_service.build_messages_for_llm(request.session_id, request.message)

    # 3. Dispatch generation to LLM provider with mapped exception handling
    try:
        reply = await provider.generate(messages, request.temperature)
    except LLMTimeoutError as err:
        logger.error("LLM timeout error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Inference request timed out: {err.message}",
        ) from err
    except LLMConnectionError as err:
        logger.error("LLM connection error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference provider unavailable: {err.message}",
        ) from err
    except LLMAuthenticationError as err:
        logger.error("LLM authentication failure: %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider authentication failed: {err.message}",
        ) from err
    except LLMModelNotFoundError as err:
        logger.error("LLM model not found: %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model unavailable: {err.message}",
        ) from err
    except LLMConfigurationError as err:
        logger.error("LLM configuration error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server provider configuration error: {err.message}",
        ) from err
    except LLMProviderError as err:
        logger.error("General LLM provider failure: %s", err)
        raise HTTPException(
            status_code=err.status_code,
            detail=f"Provider error: {err.message}",
        ) from err
    except Exception as err:
        logger.exception("Unexpected error during chat processing: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during generation.",
        ) from err

    # 4. Record new turn in conversation service
    updated_history = conv_service.record_turn(
        request.session_id, request.message, reply
    )

    return ChatResponse(
        reply=reply,
        model_used=provider.model_name,
        provider=provider.provider_name,
        history=updated_history,
    )
