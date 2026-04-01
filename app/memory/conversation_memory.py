"""
ConversationMemory — Redis-backed rolling conversation history for RAG.

Stores recent user/assistant message pairs per conversation_id,
retrievable for injection into the RAG chain prompt.
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# Redis key scheme
_KEY_CONVO = "datalens:convo:{conversation_id}"
_KEY_META = "datalens:convo_meta:{conversation_id}"

# TTL: 7 days
_DEFAULT_TTL_SECS = 7 * 24 * 3600

# Max turns to retain per conversation (rolling window)
_DEFAULT_MAX_TURNS = 20


def _make_redis_client() -> redis.Redis:
    """Create a synchronous Redis client from the configured URL."""
    return redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


class ConversationMemory:
    """
    Redis-backed conversation history store.

    Usage:
        memory = ConversationMemory()
        memory.add_message(conversation_id, "user", "What is RAG?")
        history = memory.get_history(conversation_id, limit=5)
        memory.clear(conversation_id)
    """

    def __init__(
        self,
        max_turns: int = _DEFAULT_MAX_TURNS,
        ttl_seconds: int = _DEFAULT_TTL_SECS,
    ):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = _make_redis_client()
        return self._client

    def _key(self, conversation_id: str) -> str:
        return _KEY_CONVO.format(conversation_id=conversation_id)

    def _meta_key(self, conversation_id: str) -> str:
        return _KEY_META.format(conversation_id=conversation_id)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,          # "user" | "assistant"
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Append a message to the conversation history.

        Returns the message_id (UUID).
        """
        message_id = str(uuid.uuid4())
        entry = {
            "id": message_id,
            "role": role,
            "content": content,
            "trace_id": trace_id,
            "metadata": metadata or {},
            "created_at": _now_iso(),
        }

        pipe = self.client.pipeline()
        pipe.rpush(self._key(conversation_id), json.dumps(entry))
        pipe.expire(self._key(conversation_id), self.ttl_seconds)

        # Update meta: updated_at, last_trace_id
        meta = {
            "updated_at": _now_iso(),
            "last_trace_id": trace_id,
        }
        pipe.hset(self._meta_key(conversation_id), mapping=meta)
        pipe.expire(self._meta_key(conversation_id), self.ttl_seconds)

        # Trim to max_turns pairs (2 messages per turn)
        pipe.ltrim(
            self._key(conversation_id),
            max(0, -self.max_turns * 2),
            -1,
        )
        pipe.execute()

        logger.debug(
            "Added %s message to conversation %s (id=%s)",
            role, conversation_id, message_id,
        )
        return message_id

    def add_user_message(
        self,
        conversation_id: str,
        content: str,
        trace_id: Optional[str] = None,
    ) -> str:
        return self.add_message(conversation_id, "user", content, trace_id=trace_id)

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        return self.add_message(
            conversation_id, "assistant", content,
            metadata=metadata, trace_id=trace_id,
        )

    def get_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history as a list of message dicts.

        Args:
            conversation_id: The conversation to fetch.
            limit: Max number of **message pairs** (user+assistant) to return.
                   None = return all (up to max_turns).
        """
        raw: List[str] = self.client.lrange(
            self._key(conversation_id), 0, -1
        )
        messages = [json.loads(entry) for entry in raw]

        if limit:
            # limit is pairs; each pair = 2 messages
            messages = messages[-limit * 2:]

        return messages

    def get_formatted_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> str:
        """
        Return a formatted string suitable for prompt injection:
        "User: ...\nAssistant: ...\nUser: ...\nAssistant: ..."
        """
        history = self.get_history(conversation_id, limit=limit)
        if not history:
            return ""

        lines = []
        for msg in history:
            role_label = msg["role"].capitalize()
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def clear(self, conversation_id: str) -> None:
        """Delete all history for a conversation."""
        self.client.delete(self._key(conversation_id))
        self.client.delete(self._meta_key(conversation_id))
        logger.debug("Cleared conversation %s", conversation_id)

    def get_meta(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get conversation metadata (updated_at, last_trace_id)."""
        return self.client.hgetall(self._meta_key(conversation_id)) or None

    def update_trace_id(
        self,
        conversation_id: str,
        trace_id: str,
    ) -> None:
        """Update the last trace_id on a conversation."""
        self.client.hset(
            self._meta_key(conversation_id),
            "last_trace_id",
            trace_id,
        )

    # ------------------------------------------------------------------
    # Conversation-level operations
    # ------------------------------------------------------------------

    def conversation_exists(self, conversation_id: str) -> bool:
        return self.client.exists(self._key(conversation_id)) > 0

    def get_turn_count(self, conversation_id: str) -> int:
        """Return number of user messages in the conversation."""
        history = self.get_history(conversation_id)
        return sum(1 for m in history if m["role"] == "user")


# ─────────────────────────────────────────────────────────────────
# Convenience helpers (stateless, connection-per-call for lightweight use)
# ─────────────────────────────────────────────────────────────────

_memory: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


def add_user_message(conversation_id: str, content: str, trace_id: Optional[str] = None) -> str:
    return get_conversation_memory().add_user_message(conversation_id, content, trace_id)


def add_assistant_message(
    conversation_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> str:
    return get_conversation_memory().add_assistant_message(
        conversation_id, content, metadata=metadata, trace_id=trace_id,
    )


def get_history(conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return get_conversation_memory().get_history(conversation_id, limit=limit)


def get_formatted_history(conversation_id: str, limit: Optional[int] = None) -> str:
    return get_conversation_memory().get_formatted_history(conversation_id, limit=limit)
