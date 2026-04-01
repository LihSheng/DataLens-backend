"""
Markdown exporter — renders a Conversation + its messages as a Markdown string.
"""
from datetime import datetime
from typing import Any

from app.models.conversation import Conversation, Message


def _format_timestamp(ts: datetime) -> str:
    """Return a human-readable timestamp string."""
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "Unknown"


def _role_label(role: str) -> str:
    """Return a human-readable role label."""
    return {"user": "You", "assistant": "Assistant"}.get(role, role.title())


def conversation_to_markdown(
    conversation: Conversation,
    messages: list[Message],
    include_metadata: bool = True,
) -> str:
    """
    Render a conversation as a Markdown-formatted string.

    Args:
        conversation: The Conversation ORM object.
        messages: List of Message ORM objects (ordered by created_at).
        include_metadata: Whether to include title, dates, and message counts.

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    # Title block
    lines.append(f"# {conversation.title or 'Untitled Conversation'}")
    lines.append("")

    if include_metadata:
        lines.append(f"**Created:** {_format_timestamp(conversation.created_at)}")
        lines.append(f"**Last updated:** {_format_timestamp(conversation.updated_at)}")
        lines.append(f"**Messages:** {len(messages)}")
        lines.append("")

    # Divider
    lines.append("---")
    lines.append("")

    # Messages
    for msg in messages:
        role = _role_label(msg.role)
        timestamp = _format_timestamp(msg.created_at)

        lines.append(f"### {role}  <small>{timestamp}</small>")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

        # Attach metadata JSON as collapsible detail if present
        if msg.metadata_json:
            lines.append("<details>")
            lines.append("<summary>Metadata</summary>")
            lines.append("")
            lines.append("```json")
            lines.append(msg.metadata_json)
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip()
