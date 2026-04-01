"""
Full-Text Search API — PostgreSQL-powered search across conversations and messages.

Endpoints:
- GET /api/search  — full-text search with highlighting

Uses PostgreSQL tsvector + tsquery for ranking and relevance.
Searches:
- conversations (title field)
- messages (content field)

Query params:
  q       — search query (required)
  type    — 'conversations' | 'messages' | 'all' (default: 'all')
  limit   — max results per type (default: 20, max: 100)
  offset  — pagination offset (default: 0)
"""
import logging
import re
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_query(q: str) -> str:
    """
    Strip characters that could break plainto_tsquery.
    Allow alphanumeric, spaces, and common punctuation.
    """
    return re.sub(r"[^\w\s\-'", " ", q).strip()


def _build_highlighted_snippet(text: str, query_words: list[str], max_len: int = 300) -> str:
    """
    Build a snippet with query terms wrapped in <mark> tags.
    Prioritises sentences containing query terms.
    """
    text_lower = text.lower()
    # Find first occurrence of any query word
    first_pos = -1
    for word in query_words:
        pos = text_lower.find(word.lower())
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    if first_pos == -1:
        # Fallback: start from beginning
        snippet = text[:max_len]
    else:
        # Centre snippet around the match
        start = max(0, first_pos - max_len // 2)
        end = min(len(text), start + max_len)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"

    # Wrap query words in <mark>
    for word in query_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", snippet)

    return snippet


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchMessageResult(BaseModel):
    message_id: str
    conversation_id: str
    conversation_title: str
    role: str
    content: str
    snippet: str  # highlighted excerpt
    rank: float
    created_at: str


class SearchConversationResult(BaseModel):
    conversation_id: str
    title: str
    snippet: str  # highlighted excerpt of title match
    rank: float
    created_at: str
    message_count: int


class SearchResponse(BaseModel):
    q: str
    type: str
    total_conversations: int
    total_messages: int
    conversations: list[SearchConversationResult]
    messages: list[SearchMessageResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/search", response_model=SearchResponse)
async def full_text_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    type: Literal["conversations", "messages", "all"] = Query("all"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Full-text search across conversations and messages.

    Uses PostgreSQL `plainto_tsquery('english', ...)` for safe query parsing
    and `ts_rank()` for relevance ranking. Results include highlighted snippets.
    """
    sanitised = _sanitise_query(q)
    if not sanitised:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query contains no valid characters",
        )
    query_words = sanitised.split()

    conversations: list[SearchConversationResult] = []
    messages: list[SearchMessageResult] = []
    total_conversations = 0
    total_messages = 0

    # ── Search conversations ────────────────────────────────────────────────
    if type in ("conversations", "all"):
        conv_query = text("""
            SELECT
                c.id                    AS conversation_id,
                c.title,
                c.created_at,
                COUNT(m.id)::int        AS message_count,
                ts_rank(
                    to_tsvector('english', coalesce(c.title, '')),
                    plainto_tsquery('english', :q)
                ) AS rank
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = :user_id
              AND to_tsvector('english', coalesce(c.title, ''))
                  @@ plainto_tsquery('english', :q)
            GROUP BY c.id, c.title, c.created_at
            ORDER BY rank DESC, c.created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        conv_result = await db.execute(
            conv_query,
            {"q": sanitised, "user_id": current_user.id, "limit": limit, "offset": offset},
        )
        conv_rows = conv_result.fetchall()

        count_conv_query = text("""
            SELECT COUNT(DISTINCT c.id)::int
            FROM conversations c
            WHERE c.user_id = :user_id
              AND to_tsvector('english', coalesce(c.title, ''))
                  @@ plainto_tsquery('english', :q)
        """)
        count_result = await db.execute(count_conv_query, {"q": sanitised, "user_id": current_user.id})
        total_conversations = count_result.scalar() or 0

        for row in conv_rows:
            conversations.append(SearchConversationResult(
                conversation_id=row.conversation_id,
                title=row.title or "Untitled",
                snippet=_build_highlighted_snippet(row.title or "", query_words),
                rank=float(row.rank) if row.rank else 0.0,
                created_at=str(row.created_at) if row.created_at else "",
                message_count=row.message_count,
            ))

    # ── Search messages ─────────────────────────────────────────────────────
    if type in ("messages", "all"):
        msg_query = text("""
            SELECT
                m.id                    AS message_id,
                m.conversation_id,
                c.title                AS conversation_title,
                m.role,
                m.content,
                m.created_at,
                ts_rank(
                    to_tsvector('english', coalesce(m.content, '')),
                    plainto_tsquery('english', :q)
                ) AS rank
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.user_id = :user_id
              AND to_tsvector('english', coalesce(m.content, ''))
                  @@ plainto_tsquery('english', :q)
            ORDER BY rank DESC, m.created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        msg_result = await db.execute(
            msg_query,
            {"q": sanitised, "user_id": current_user.id, "limit": limit, "offset": offset},
        )
        msg_rows = msg_result.fetchall()

        count_msg_query = text("""
            SELECT COUNT(m.id)::int
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.user_id = :user_id
              AND to_tsvector('english', coalesce(m.content, ''))
                  @@ plainto_tsquery('english', :q)
        """)
        count_result = await db.execute(count_msg_query, {"q": sanitised, "user_id": current_user.id})
        total_messages = count_result.scalar() or 0

        for row in msg_rows:
            messages.append(SearchMessageResult(
                message_id=row.message_id,
                conversation_id=row.conversation_id,
                conversation_title=row.conversation_title or "Untitled",
                role=row.role,
                content=row.content,
                snippet=_build_highlighted_snippet(row.content or "", query_words),
                rank=float(row.rank) if row.rank else 0.0,
                created_at=str(row.created_at) if row.created_at else "",
            ))

    return SearchResponse(
        q=q,
        type=type,
        total_conversations=total_conversations,
        total_messages=total_messages,
        conversations=conversations,
        messages=messages,
    )
