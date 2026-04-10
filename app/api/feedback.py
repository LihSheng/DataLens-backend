"""
Feedback API — Stage 6.

POST /api/feedback          — record feedback on a RAG answer
GET  /api/feedback/{message_id}  — get feedback for a specific message
GET  /api/feedback/conversation/{conversation_id}  — list all feedback in a conversation

Stores votes (positive/negative), 1-5 ratings, and optional comments in PostgreSQL.
"""
import json
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.feedback import Feedback
from app.models.conversation import Message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    message_id: str
    user_id: str
    vote: Optional[str] = Field(None, description="'positive' or 'negative'")
    rating: Optional[int] = Field(None, ge=1, le=5, description="1-5 star rating")
    comment: Optional[str] = None
    metadata: Optional[dict] = None


class FeedbackResponse(BaseModel):
    id: str
    message_id: str
    user_id: str
    vote: Optional[str]
    rating: Optional[int]
    comment: Optional[str]
    metadata: Optional[dict]
    created_at: str

    class Config:
        from_attributes = True


class FeedbackListResponse(BaseModel):
    items: List[FeedbackResponse]
    total: int


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _feedback_to_response(fb: Feedback) -> FeedbackResponse:
    metadata = None
    if fb.metadata_json:
        try:
            metadata = json.loads(fb.metadata_json)
        except Exception:
            metadata = {"raw": fb.metadata_json}

    return FeedbackResponse(
        id=fb.id,
        message_id=fb.message_id,
        user_id=fb.user_id,
        vote=fb.vote,
        rating=fb.rating,
        comment=fb.comment,
        metadata=metadata,
        created_at=fb.created_at.isoformat() if fb.created_at else "",
    )


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("", response_model=FeedbackResponse, status_code=201)
async def create_feedback(
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Record user feedback (thumbs up/down, rating, comment) on a RAG answer.

    Stores a new feedback row linked to the message_id.
    """
    # Validate vote value
    if payload.vote is not None and payload.vote not in ("positive", "negative"):
        raise HTTPException(400, "vote must be 'positive' or 'negative'")

    metadata_json = json.dumps(payload.metadata) if payload.metadata else None

    feedback = Feedback(
        message_id=payload.message_id,
        user_id=payload.user_id,
        vote=payload.vote,
        rating=payload.rating,
        comment=payload.comment,
        metadata_json=metadata_json,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return _feedback_to_response(feedback)


@router.get("/{message_id}", response_model=FeedbackListResponse)
async def get_feedback_for_message(
    message_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all feedback entries for a specific message.
    """
    result = await db.execute(
        select(Feedback)
        .where(Feedback.message_id == message_id)
        .order_by(Feedback.created_at.desc())
    )
    rows = result.scalars().all()

    return FeedbackListResponse(
        items=[_feedback_to_response(fb) for fb in rows],
        total=len(rows),
    )


@router.get("/conversation/{conversation_id}", response_model=FeedbackListResponse)
async def get_feedback_for_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    List all feedback entries for all messages in a conversation.

    Joins messages → feedback to retrieve feedback associated with
    every message belonging to the given conversation.
    """
    result = await db.execute(
        select(Feedback)
        .join(Message, Feedback.message_id == Message.id)
        .where(Message.conversation_id == conversation_id)
        .order_by(Feedback.created_at.desc())
    )
    rows = result.scalars().all()

    return FeedbackListResponse(
        items=[_feedback_to_response(fb) for fb in rows],
        total=len(rows),
    )


# ─────────────────────────────────────────────────────────
# Observability stats — called by the observability UI
# ─────────────────────────────────────────────────────────

class FeedbackStatsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    positiveRatio: float
    negativeRatio: float


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate feedback counts across all messages.
    Returns total, positive, negative, and ratios.
    """
    result = await db.execute(
        select(Feedback.vote, func.count(Feedback.id))
        .group_by(Feedback.vote)
    )
    rows = result.all()

    counts: dict[str, int] = {"positive": 0, "negative": 0}
    for vote, count in rows:
        if vote in counts:
            counts[vote] = count

    total = counts["positive"] + counts["negative"]
    positive_ratio = counts["positive"] / total if total > 0 else 0.0
    negative_ratio = counts["negative"] / total if total > 0 else 0.0

    return FeedbackStatsResponse(
        total=total,
        positive=counts["positive"],
        negative=counts["negative"],
        positiveRatio=round(positive_ratio, 4),
        negativeRatio=round(negative_ratio, 4),
    )
