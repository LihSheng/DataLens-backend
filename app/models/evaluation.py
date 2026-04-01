"""RAGSettings model — per-user RAG behaviour flags."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Boolean, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class RAGSettings(Base):
    __tablename__ = "rag_settings"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    query_expansion: Mapped[bool] = mapped_column(Boolean, default=False)
    hyde: Mapped[bool] = mapped_column(Boolean, default=False)
    reranker: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
