"""GoldenDataset model — Stage 6."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class GoldenDataset(Base):
    __tablename__ = "golden_dataset"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    question: Mapped[str] = mapped_column(String, nullable=False)
    answer: Mapped[str] = mapped_column(String, nullable=False)
    context: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # 'manual' | 'extracted'
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
