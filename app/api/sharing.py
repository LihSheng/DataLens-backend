"""
Conversation Sharing API — generate and manage read-only share links.

Endpoints:
- POST   /api/conversations/{conversation_id}/share          — create share token
- GET    /api/conversations/{conversation_id}/share           — get active share link
- DELETE /api/conversations/{conversation_id}/share           — revoke share link
- GET    /api/shared/{token}                                  — public read-only view
- POST   /api/shared/{token}/view                             — increment view count
"""
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.share_token import ShareToken
from app.export import conversation_to_markdown

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/conversations", tags=["sharing"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class ShareLinkResponse(BaseModel):
    share_url: str
    token: str
    expires_at: Optional[datetime]
    is_active: bool


class ShareTokenResponse(BaseModel):
    id: str
    conversation_id: str
    token: str
    created_by: str
    expires_at: Optional[datetime]
    is_active: bool
    view_count: int
    created_at: datetime


class SharedConversationResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    messages: list[dict]


class ViewCountResponse(BaseModel):
    token: str
    view_count: int


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_share_url(token: str) -> str:
    """Build the public share URL for a token."""
    return f"/shared/{token}"


def _generate_token() -> str:
    """Generate a cryptographically secure share token."""
    return secrets.token_urlsafe(32)


async def _get_owned_conversation(
    conversation_id: str,
    user_id: str,
    db: AsyncSession,
) -> Conversation:
    """Load a conversation verifying the current user owns it."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == user_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conv


# ── Authenticated share management ─────────────────────────────────────────────

@router.post(
    "/{conversation_id}/share",
    response_model=ShareTokenResponse,
    summary="Create share link",
)
async def create_share_link(
    conversation_id: str,
    expires_days: Optional[int] = None,  # None = never expires
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a new share token for a conversation.

    If an active share token already exists, returns it instead of creating a duplicate.
    Set expires_days to auto-expire the link (e.g., 7 for 7 days).
    """
    # Verify ownership
    await _get_owned_conversation(conversation_id, current_user.id, db)

    # Check for existing active token
    result = await db.execute(
        select(ShareToken)
        .where(ShareToken.conversation_id == conversation_id)
        .where(ShareToken.is_active == True)  # noqa: E712
    )
    existing = result.scalar_one_or_none()
    if existing:
        return ShareTokenResponse.model_validate(existing)

    # Create new token
    token = _generate_token()
    expires_at = None
    if expires_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_days)

    share_token = ShareToken(
        conversation_id=conversation_id,
        token=token,
        created_by=current_user.id,
        expires_at=expires_at,
        is_active=True,
    )
    db.add(share_token)
    await db.commit()
    await db.refresh(share_token)

    return ShareTokenResponse.model_validate(share_token)


@router.get(
    "/{conversation_id}/share",
    response_model=ShareTokenResponse,
    summary="Get active share link",
)
async def get_share_link(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the active share token for a conversation, if any."""
    await _get_owned_conversation(conversation_id, current_user.id, db)

    result = await db.execute(
        select(ShareToken)
        .where(ShareToken.conversation_id == conversation_id)
        .where(ShareToken.is_active == True)  # noqa: E712
    )
    token = result.scalar_one_or_none()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active share link for this conversation",
        )

    return ShareTokenResponse.model_validate(token)


@router.delete(
    "/{conversation_id}/share",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke share link",
)
async def revoke_share_link(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke (deactivate) the active share token for a conversation."""
    await _get_owned_conversation(conversation_id, current_user.id, db)

    result = await db.execute(
        select(ShareToken)
        .where(ShareToken.conversation_id == conversation_id)
        .where(ShareToken.is_active == True)  # noqa: E712
    )
    token = result.scalar_one_or_none()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active share link to revoke",
        )

    token.is_active = False
    await db.commit()
    return None


# ── Public read-only share view ────────────────────────────────────────────────

@router.get(
    "/shared/{token}",
    response_model=SharedConversationResponse,
    summary="View shared conversation",
)
async def view_shared_conversation(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Public (no auth required) endpoint to retrieve a shared conversation.

    Returns the conversation title and all messages. Expired or inactive
    tokens return 404.
    """
    result = await db.execute(
        select(ShareToken)
        .where(ShareToken.token == token)
        .where(ShareToken.is_active == True)  # noqa: E712
    )
    share_token = result.scalar_one_or_none()
    if not share_token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or inactive",
        )

    if share_token.expires_at and share_token.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has expired",
        )

    # Load conversation + messages
    conv_result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == share_token.conversation_id)
    )
    conv = conv_result.scalar_one_or_none()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = sorted(conv.messages, key=lambda m: m.created_at)

    return SharedConversationResponse(
        conversation_id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        messages=[
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
                "metadata": m.metadata_json,
            }
            for m in messages
        ],
    )


@router.post(
    "/shared/{token}/view",
    response_model=ViewCountResponse,
    summary="Record a view on shared conversation",
)
async def record_shared_view(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Increment the view counter for a shared conversation.

    Called by the frontend when a shared page is opened.
    Returns the updated view count.
    """
    result = await db.execute(
        select(ShareToken)
        .where(ShareToken.token == token)
        .where(ShareToken.is_active == True)  # noqa: E712
    )
    share_token = result.scalar_one_or_none()
    if not share_token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found or inactive",
        )

    if share_token.expires_at and share_token.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has expired",
        )

    share_token.view_count = (share_token.view_count or 0) + 1
    await db.commit()
    await db.refresh(share_token)

    return ViewCountResponse(token=token, view_count=share_token.view_count)
