"""Query cost estimation and persistence helpers."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import QueryCost

logger = logging.getLogger(__name__)

# USD per 1K tokens (input/output)
MODEL_PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.00500, "output": 0.01500},
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (input_tokens / 1000.0) * pricing["input"]
        + (output_tokens / 1000.0) * pricing["output"],
        8,
    )


async def _ensure_query_cost_table(db: AsyncSession) -> None:
    # Safety net for environments running without migrations.
    await db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS query_costs (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                conversation_id VARCHAR NOT NULL,
                trace_id VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_usd FLOAT DEFAULT 0,
                timestamp TIMESTAMP
            )
            """
        )
    )


async def record_query_cost(
    *,
    db: AsyncSession,
    user_id: str,
    conversation_id: str,
    trace_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Optional[float] = None,
) -> QueryCost:
    if cost_usd is None:
        cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)

    try:
        await _ensure_query_cost_table(db)
        row = QueryCost(
            user_id=user_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            timestamp=datetime.utcnow(),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row
    except Exception as exc:
        await db.rollback()
        logger.warning("Failed to persist QueryCost: %s", exc)
        raise

