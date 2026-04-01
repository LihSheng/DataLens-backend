"""Retention policy model — Stage 7 (Governance)."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Boolean, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class RetentionPolicy(Base):
    """
    Configurable retention policy for data lifecycle management.

    Controls how long conversations, messages, and feedback are kept
    before automatic cleanup. Each resource type has its own policy.
    """
    __tablename__ = "retention_policy"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Resource type this policy applies to
    resource: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False
    )  # 'conversations' | 'messages' | 'feedback'
    # Retention period
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False, default=90)
    # Whether this policy is active
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Audit trail
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
