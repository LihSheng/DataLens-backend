"""RAGSettings model — per-user RAG behaviour flags."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Boolean, Float, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class RAGSettings(Base):
    __tablename__ = "rag_settings"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    # Retrieval options
    query_expansion: Mapped[bool] = mapped_column(Boolean, default=False)
    hyde: Mapped[bool] = mapped_column(Boolean, default=False)
    reranker: Mapped[bool] = mapped_column(Boolean, default=False)

    # Safety + quality (Stage 3)
    guardrails_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    prompt_injection_check: Mapped[bool] = mapped_column(Boolean, default=True)
    grounding_check: Mapped[bool] = mapped_column(Boolean, default=True)
    citation_validation: Mapped[bool] = mapped_column(Boolean, default=True)

    # Thresholds
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    grounding_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    injection_confidence_threshold: Mapped[float] = mapped_column(Float, default=0.7)

    # Citation
    required_citation_threshold: Mapped[float] = mapped_column(Float, default=0.5)

    # Retry / fallback
    max_retries: Mapped[int] = mapped_column(Integer, default=1)

    # Memory (Stage 4)
    enable_memory: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_followup: Mapped[bool] = mapped_column(Boolean, default=True)
    conversation_history_limit: Mapped[int] = mapped_column(Integer, default=10)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
