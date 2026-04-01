"""
Golden Dataset API — Stage 6.

POST /api/golden            — add a golden Q&A pair
GET  /api/golden            — list golden dataset (with search/filter/pagination)
DELETE /api/golden/{id}     — delete a golden example
GET  /api/golden/{id}       — get a single golden example

The golden dataset serves as ground truth for RAG evaluation.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.evaluation.golden_dataset import (
    create_golden_entry,
    list_golden_entries,
    get_golden_entry,
    delete_golden_entry,
    GoldenDatasetCreate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/golden", tags=["golden_dataset"])


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class GoldenCreate(BaseModel):
    question: str = Field(..., min_length=1, description="The golden question")
    answer: str = Field(..., min_length=1, description="The expected answer")
    context: Optional[List[str]] = Field(
        default=None, description="Source context texts"
    )
    source: Optional[str] = Field(
        default="manual",
        description="Origin: 'manual', 'extracted', etc.",
    )
    tags: Optional[List[str]] = Field(default=None, description="Arbitrary tags")
    created_by: Optional[str] = Field(default=None, description="User ID")


class GoldenResponse(BaseModel):
    id: str
    question: str
    answer: str
    context: Optional[List[str]]
    source: Optional[str]
    tags: Optional[List[str]]
    created_by: Optional[str]
    created_at: str


class GoldenListResponse(BaseModel):
    items: List[GoldenResponse]
    total: int
    limit: int
    offset: int


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("", response_model=GoldenResponse, status_code=201)
async def add_golden_entry(
    payload: GoldenCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Add a new golden Q&A pair to the dataset.
    """
    data = GoldenDatasetCreate(
        question=payload.question,
        answer=payload.answer,
        context=payload.context,
        source=payload.source or "manual",
        tags=payload.tags,
        created_by=payload.created_by,
    )
    row = await create_golden_entry(db, data)
    return GoldenResponse(**row.to_dict())


@router.get("", response_model=GoldenListResponse)
async def list_golden(
    search: Optional[str] = Query(None, description="Full-text search over question+answer"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by"),
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    List golden dataset entries with optional search and pagination.
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    rows, total = await list_golden_entries(
        db,
        search=search,
        tags=tag_list,
        source=source,
        limit=limit,
        offset=offset,
    )

    return GoldenListResponse(
        items=[GoldenResponse(**r.to_dict()) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{entry_id}", response_model=GoldenResponse)
async def get_golden(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a single golden dataset entry by ID.
    """
    row = await get_golden_entry(db, entry_id)
    if row is None:
        raise HTTPException(404, "Golden entry not found")
    return GoldenResponse(**row.to_dict())


@router.delete("/{entry_id}", status_code=204)
async def delete_golden(
    entry_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a golden dataset entry by ID.
    """
    deleted = await delete_golden_entry(db, entry_id)
    if not deleted:
        raise HTTPException(404, "Golden entry not found")
    return None
