"""Document model with versioning fields."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(Integer, default=0)
    extension: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="processing")
    parse_error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ocr_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    pii_entities_found: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # JSON list
    chunking_strategy: Mapped[str] = mapped_column(String, default="recursive")
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_document_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("documents.id"), nullable=True
    )
    is_active_version: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DocumentAcl(Base):
    __tablename__ = "document_acl"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), nullable=False
    )
    principal_type: Mapped[str] = mapped_column(String)  # "user" | "role"
    principal_id: Mapped[str] = mapped_column(String)
    can_read: Mapped[bool] = mapped_column(Boolean, default=True)
