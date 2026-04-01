"""MessageFeedback model."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    message_id: Mapped[str] = mapped_column(String)
    conversation_id: Mapped[str] = mapped_column(String)
    trace_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rating: Mapped[str] = mapped_column(String)  # "positive" | "negative"
    comment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
