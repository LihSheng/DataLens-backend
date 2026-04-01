"""AuditEvent, QueryCost, and GoldenQuestion models."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_type: Mapped[str] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String)
    user_email: Mapped[str] = mapped_column(String)
    ip_address: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payload_json: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QueryCost(Base):
    __tablename__ = "query_costs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String)
    conversation_id: Mapped[str] = mapped_column(String)
    trace_id: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GoldenQuestion(Base):
    __tablename__ = "golden_questions"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    question: Mapped[str] = mapped_column(String)
    expected_answer: Mapped[str] = mapped_column(String)
    relevant_document_ids: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # JSON list
    min_faithfulness: Mapped[float] = mapped_column(default=0.8)
    min_relevance: Mapped[float] = mapped_column(default=0.75)
    last_faithfulness: Mapped[Optional[float]] = mapped_column(nullable=True)
    last_relevance: Mapped[Optional[float]] = mapped_column(nullable=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
