"""
Phoenix Proxy — Stage 4 Security Layer.

All Phoenix API calls are routed through this backend proxy so that
Phoenix is never exposed directly to the frontend. Every endpoint
requires admin authentication.
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any

from app.config import settings
from app.dependencies import require_admin

router = APIRouter(prefix="/api/phoenix", tags=["phoenix"])

# Reusable HTTP client for Phoenix.  Timeout is generous for eval queries.
_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.phoenix_base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    return _client


async def _get(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET helper that returns raw JSON from Phoenix or raises a JSON error."""
    client = await _get_client()
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Phoenix request failed: {exc}",
        )


# ─────────────────────────────────────────────────────────
# Proxy endpoints
# ─────────────────────────────────────────────────────────

@router.get("/traces")
async def list_traces(
    limit: int = 100,
    offset: int = 0,
    _: None = Depends(require_admin),
) -> Any:
    """
    Proxy GET /api/phoenix/traces?limit=100&offset=0
    """
    return await _get("/api/traces", params={"limit": limit, "offset": offset})


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    _: None = Depends(require_admin),
) -> Any:
    """
    Proxy GET /api/phoenix/traces/{trace_id}
    """
    return await _get(f"/api/traces/{trace_id}")


@router.get("/spans")
async def list_spans(
    trace_id: str,
    _: None = Depends(require_admin),
) -> Any:
    """
    Proxy GET /api/phoenix/spans?trace_id={trace_id}
    """
    return await _get("/api/spans", params={"trace_id": trace_id})


@router.get("/evaluations")
async def list_evaluations(
    trace_id: str,
    _: None = Depends(require_admin),
) -> Any:
    """
    Proxy GET /api/phoenix/evaluations?trace_id={trace_id}
    """
    return await _get("/api/evaluations", params={"trace_id": trace_id})


@router.get("/summary")
async def get_summary(
    _: None = Depends(require_admin),
) -> Any:
    """
    Proxy GET /api/phoenix/summary
    """
    return await _get("/api/summary")
