"""Conversation service managing state, validation, prompt assembly, and memory limits."""

from collections import deque
import logging
from pathlib import Path
import threading
from typing import Optional

from app.config import Settings, get_settings
from app.models import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer concisely and clearly. "
    "If you don't know something, say so."
)


class ConversationService:
    """Manages multi-turn conversation sessions, validation, and bounded memory."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.max_history = self.settings.max_history_messages
        self._store: dict[str, deque[ChatMessage]] = {}
        self._lock = threading.Lock()
        self._system_prompt = self._load_system_prompt(self.settings.system_prompt_path)

    def _load_system_prompt(self, path: Path) -> str:
        """Load system prompt from file with safe fallback."""
        try:
            if path.is_file():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    logger.info("Loaded system prompt from %s", path)
                    return content
        except Exception as err:
            logger.warning("Failed to load system prompt from %s: %s. Using default.", path, err)
        return DEFAULT_SYSTEM_PROMPT

    @property
    def system_prompt(self) -> str:
        """Current system prompt."""
        return self._system_prompt

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve a copy of bounded conversation history for a session."""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = deque(maxlen=self.max_history)
            return list(self._store[session_id])

    def sync_client_history(
        self, session_id: str, client_history: Optional[list[ChatMessage]]
    ) -> list[ChatMessage]:
        """Safely validate, truncate, and synchronize history provided by client."""
        if client_history is None:
            return self.get_history(session_id)

        validated: list[ChatMessage] = []
        for msg in client_history:
            # Strictly filter roles: only user and assistant are allowed in multi-turn history
            if msg.role in (MessageRole.USER.value, MessageRole.ASSISTANT.value):
                content = msg.content.strip()
                if content:
                    # Enforce per-message maximum length of 4000 chars to avoid memory abuse
                    truncated_content = content[:4000]
                    validated.append(ChatMessage(role=msg.role, content=truncated_content))

        # Enforce server-side memory bounds
        bounded = validated[-self.max_history:]

        with self._lock:
            self._store[session_id] = deque(bounded, maxlen=self.max_history)
            return list(self._store[session_id])

    def build_messages_for_llm(
        self, session_id: str, new_user_message: str
    ) -> list[ChatMessage]:
        """Construct prompt message sequence: system prompt + recent history + current user message."""
        history = self.get_history(session_id)
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM.value, content=self._system_prompt)
        ]
        for msg in history:
            messages.append(ChatMessage(role=msg.role, content=msg.content))
        messages.append(ChatMessage(role=MessageRole.USER.value, content=new_user_message))
        return messages

    def record_turn(self, session_id: str, user_message: str, assistant_reply: str) -> list[ChatMessage]:
        """Append user message and assistant reply to session memory, preserving bounds."""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = deque(maxlen=self.max_history)
            dq = self._store[session_id]
            dq.append(ChatMessage(role=MessageRole.USER.value, content=user_message.strip()))
            dq.append(ChatMessage(role=MessageRole.ASSISTANT.value, content=assistant_reply.strip()))
            return list(dq)

    def clear_session(self, session_id: str) -> None:
        """Clear memory for a given session."""
        with self._lock:
            if session_id in self._store:
                self._store[session_id].clear()
