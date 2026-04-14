"""User block log model — audit trail for block/unblock actions."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class UserBlockLog(Base):
    """
    Immutable audit trail for user block/unblock actions.

    Records who blocked/unblocked whom, when, and why.
    Used for compliance and security auditing.
    """
    __tablename__ = "user_block_logs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # "blocked" | "unblocked"
    performed_by: Mapped[str] = mapped_column(
        String, ForeignKey("users.id"), nullable=False, index=True
    )
    reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
