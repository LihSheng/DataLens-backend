"""
Golden dataset management — Stage 6.

Provides CRUD operations and search for the golden question/answer dataset.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.golden_dataset import GoldenDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Schemas / DTOs
# ─────────────────────────────────────────────────────────

class GoldenDatasetCreate:
    """Input schema for creating a golden dataset entry."""

    def __init__(
        self,
        question: str,
        answer: str,
        context: Optional[List[str]] = None,
        source: Optional[str] = "manual",
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ):
        self.question = question
        self.answer = answer
        self.context = context or []
        self.source = source
        self.tags = tags or []
        self.created_by = created_by


class GoldenDatasetRow:
    """Output schema for a golden dataset entry."""

    def __init__(
        self,
        id: str,
        question: str,
        answer: str,
        context: Optional[List[str]],
        source: Optional[str],
        tags: Optional[List[str]],
        created_by: Optional[str],
        created_at: Any,
    ):
        self.id = id
        self.question = question
        self.answer = answer
        self.context = context
        self.source = source
        self.tags = tags
        self.created_by = created_by
        self.created_at = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "source": self.source,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────────────────
# CRUD operations
# ─────────────────────────────────────────────────────────

async def create_golden_entry(
    db: AsyncSession,
    data: GoldenDatasetCreate,
) -> GoldenDatasetRow:
    """
    Add a new golden Q&A entry to the dataset.

    Args:
        db: Async SQLAlchemy session.
        data: GoldenDatasetCreate with question, answer, context, source, tags, created_by.

    Returns:
        GoldenDatasetRow of the created entry.
    """
    entry = GoldenDataset(
        question=data.question,
        answer=data.answer,
        context=data.context,
        source=data.source,
        tags=data.tags,
        created_by=data.created_by,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return _to_row(entry)


async def list_golden_entries(
    db: AsyncSession,
    search: Optional[str] = None,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[List[GoldenDatasetRow], int]:
    """
    List golden dataset entries with optional search and filters.

    Args:
        db: Async SQLAlchemy session.
        search: Free-text search over question + answer.
        tags: Filter by tags (any match).
        source: Filter by source.
        limit: Max rows to return.
        offset: Rows to skip.

    Returns:
        (list of rows, total count).
    """
    stmt = select(GoldenDataset)
    count_stmt = select(func.count(GoldenDataset.id))

    if search:
        search_filter = or_(
            GoldenDataset.question.ilike(f"%{search}%"),
            GoldenDataset.answer.ilike(f"%{search}%"),
        )
        stmt = stmt.where(search_filter)
        count_stmt = count_stmt.where(search_filter)

    if source:
        stmt = stmt.where(GoldenDataset.source == source)
        count_stmt = count_stmt.where(GoldenDataset.source == source)

    if tags:
        # Match any of the provided tags
        stmt = stmt.where(GoldenDataset.tags.overlap(tags))
        count_stmt = count_stmt.where(GoldenDataset.tags.overlap(tags))

    # Order by created_at desc
    stmt = stmt.order_by(GoldenDataset.created_at.desc()).offset(offset).limit(limit)

    results = await db.execute(stmt)
    entries = results.scalars().all()

    count_result = await db.execute(count_stmt)
    total = count_result.scalar() or 0

    return [_to_row(e) for e in entries], total


async def get_golden_entry(
    db: AsyncSession,
    entry_id: str,
) -> Optional[GoldenDatasetRow]:
    """Get a single golden entry by ID."""
    result = await db.execute(
        select(GoldenDataset).where(GoldenDataset.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        return None
    return _to_row(entry)


async def delete_golden_entry(
    db: AsyncSession,
    entry_id: str,
) -> bool:
    """
    Delete a golden dataset entry by ID.

    Returns:
        True if deleted, False if not found.
    """
    result = await db.execute(
        select(GoldenDataset).where(GoldenDataset.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        return False
    await db.delete(entry)
    await db.commit()
    return True


async def get_all_golden_entries(
    db: AsyncSession,
) -> List[GoldenDatasetRow]:
    """Get all golden entries (for experiment runs)."""
    result = await db.execute(
        select(GoldenDataset).order_by(GoldenDataset.created_at.desc())
    )
    entries = result.scalars().all()
    return [_to_row(e) for e in entries]


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _to_row(entry: GoldenDataset) -> GoldenDatasetRow:
    return GoldenDatasetRow(
        id=entry.id,
        question=entry.question,
        answer=entry.answer,
        context=entry.context,
        source=entry.source,
        tags=entry.tags,
        created_by=entry.created_by,
        created_at=entry.created_at,
    )
