"""Feedback model — Stage 6."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    message_id: Mapped[str] = mapped_column(
        String, nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String, nullable=False, index=True
    )
    vote: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 'positive' | 'negative'
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5
    comment: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # JSON blob
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
