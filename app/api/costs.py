"""Costs API for Stage 5 query-cost observability."""
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.audit import QueryCost

router = APIRouter()


class CostRow(BaseModel):
    id: str
    conversation_id: str
    trace_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostSummary(BaseModel):
    total_queries: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


@router.get("/costs/recent", response_model=List[CostRow])
async def list_recent_costs(
    limit: int = Query(20, ge=1, le=200),
    conversation_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(QueryCost).order_by(QueryCost.timestamp.desc()).limit(limit)
    if conversation_id:
        stmt = stmt.where(QueryCost.conversation_id == conversation_id)
    rows = (await db.execute(stmt)).scalars().all()
    return [
        CostRow(
            id=row.id,
            conversation_id=row.conversation_id,
            trace_id=row.trace_id,
            model=row.model,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cost_usd=row.cost_usd,
        )
        for row in rows
    ]


@router.get("/costs/summary", response_model=CostSummary)
async def get_cost_summary(
    conversation_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(
        func.count(QueryCost.id),
        func.coalesce(func.sum(QueryCost.input_tokens), 0),
        func.coalesce(func.sum(QueryCost.output_tokens), 0),
        func.coalesce(func.sum(QueryCost.cost_usd), 0.0),
    )
    if conversation_id:
        stmt = stmt.where(QueryCost.conversation_id == conversation_id)
    total_queries, input_tokens, output_tokens, total_cost = (await db.execute(stmt)).one()
    return CostSummary(
        total_queries=int(total_queries or 0),
        total_input_tokens=int(input_tokens or 0),
        total_output_tokens=int(output_tokens or 0),
        total_cost_usd=float(total_cost or 0.0),
    )
