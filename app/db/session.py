"""
Stub async SQLAlchemy session.
Full implementation uses asyncpg; placeholder here until db layer is wired.
"""
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def create_tables():
    """Create all tables — called on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_sqlite_schema(conn)


async def _ensure_sqlite_schema(conn) -> None:
    """
    Lightweight SQLite schema evolution for local dev.
    Production should use proper migrations (Postgres + SQL migrations/Alembic).
    """
    if not str(settings.database_url).startswith("sqlite"):
        return

    try:
        rows = (await conn.exec_driver_sql("PRAGMA table_info(documents)")).fetchall()
        existing = {r[1] for r in rows}  # (cid, name, type, notnull, dflt_value, pk)
        if "chunk_count" not in existing:
            await conn.exec_driver_sql(
                "ALTER TABLE documents ADD COLUMN chunk_count INTEGER DEFAULT 0"
            )
    except Exception:
        # Never block app startup due to dev-only migration helpers.
        return


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency-injectable database session."""
    async with AsyncSessionLocal() as session:
        yield session
