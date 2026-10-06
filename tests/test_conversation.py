"""Tests for conversation memory bounds, validation, and session management."""

from app.config import Settings
from app.models import ChatMessage
from app.services.conversation import ConversationService


def test_conversation_history_bounding():
    """Verify that conversation history is strictly capped at max_history_messages."""
    settings = Settings(max_history_messages=4)
    service = ConversationService(settings)
    session_id = "bounded_test_session"

    # Add 4 turns (8 messages)
    for i in range(4):
        service.record_turn(session_id, f"User message {i}", f"Assistant reply {i}")

    history = service.get_history(session_id)
    # deque maxlen is 4, so history should have at most 4 items
    assert len(history) == 4
    # The last two items should be from the final turn
    assert history[-2].content == "User message 3"
    assert history[-1].content == "Assistant reply 3"


def test_system_prompt_prepended_to_messages():
    """Verify that system prompt is always injected as the first message."""
    settings = Settings(max_history_messages=6)
    service = ConversationService(settings)
    session_id = "prompt_assembly_session"

    service.record_turn(session_id, "Previous question", "Previous answer")
    messages = service.build_messages_for_llm(session_id, "Latest question")

    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[0].content == service.system_prompt
    assert messages[1].role == "user"
    assert messages[1].content == "Previous question"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Previous answer"
    assert messages[3].role == "user"
    assert messages[3].content == "Latest question"


def test_client_history_synchronization_and_filtering():
    """Verify client history synchronization strips extra messages beyond max_history."""
    settings = Settings(max_history_messages=3)
    service = ConversationService(settings)
    session_id = "sync_session"

    client_history = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Reply 1"),
        ChatMessage(role="user", content="Turn 2"),
        ChatMessage(role="assistant", content="Reply 2"),
        ChatMessage(role="user", content="Turn 3"),
    ]

    synced = service.sync_client_history(session_id, client_history)
    assert len(synced) == 3
    # Should retain the last 3 messages
    assert synced[0].content == "Turn 2"
    assert synced[1].content == "Reply 2"
    assert synced[2].content == "Turn 3"


def test_session_isolation():
    """Verify that different session IDs maintain separate conversation memories."""
    settings = Settings(max_history_messages=5)
    service = ConversationService(settings)

    service.record_turn("session_alpha", "Alpha question", "Alpha reply")
    service.record_turn("session_beta", "Beta question", "Beta reply")

    alpha_history = service.get_history("session_alpha")
    beta_history = service.get_history("session_beta")

    assert len(alpha_history) == 2
    assert alpha_history[0].content == "Alpha question"

    assert len(beta_history) == 2
    assert beta_history[0].content == "Beta question"


def test_clear_session():
    """Verify that clearing session history removes stored messages."""
    settings = Settings(max_history_messages=5)
    service = ConversationService(settings)
    session_id = "session_to_clear"

    service.record_turn(session_id, "Hello", "Hi there")
    assert len(service.get_history(session_id)) == 2

    service.clear_session(session_id)
    assert len(service.get_history(session_id)) == 0
