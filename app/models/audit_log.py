"""Audit log model — Stage 7 (Governance)."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AuditLog(Base):
    """
    Immutable audit trail for all admin actions.

    Records who did what, when, and from where.
    Used for compliance, security auditing, and accountability.
    """
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Who
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # What
    action: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # e.g. 'create_user', 'update_settings', 'delete_experiment'
    resource: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # e.g. 'user', 'experiment', 'settings'
    resource_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    # Extra context
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Where
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # When
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
