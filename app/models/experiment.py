"""Experiment models — Stage 6."""
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    config_json: Mapped[str] = mapped_column(String, nullable=False)  # JSON config
    results_json: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # aggregated scores
    status: Mapped[str] = mapped_column(String, default="pending")  # pending | running | completed | failed
    created_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    results: Mapped[List["ExperimentResult"]] = relationship(
        "ExperimentResult",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    question: Mapped[str] = mapped_column(String, nullable=False)
    expected_answer: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    generated_answer: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    retrieved_contexts: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String), nullable=True
    )
    metrics_json: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # RAGAS scores
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship(
        back_populates="results",
    )
