"""ConnectorConfig model — persistent configuration for external data connectors."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ConnectorConfig(Base):
    __tablename__ = "connectors_config"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    connector_type: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False
    )  # 'filesystem', 's3', 'googledrive', 'notion'
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)  # connection credentials, paths
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
