"""
Health + Readiness endpoints.

GET  /api/health  — liveness probe (always 200 if app is running)
GET  /api/ready   — readiness probe (checks DB, Redis, vector store)
"""
from typing import Optional

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "not_ready"
    checks: dict[str, str]
    message: Optional[str] = None


# ─── Liveness ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(status="ok")


# ─── Readiness ─────────────────────────────────────────────────────────────

async def _check_database() -> tuple[bool, str]:
    """Check if database is reachable."""
    try:
        from sqlalchemy import text
        from app.db.session import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _check_redis() -> tuple[bool, str]:
    """Check if Redis is reachable (sync client)."""
    try:
        import redis
        from app.config import settings as app_settings

        client = redis.from_url(
            app_settings.redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        client.ping()
        client.close()
        return True, "ok"
    except Exception as e:
        return False, str(e)


async def _check_vectorstore() -> tuple[bool, str]:
    """Check if vector store is initialized."""
    try:
        from app.services.vectorstore_service import get_vectorstore

        vs = get_vectorstore()
        if vs is None:
            return False, "vectorstore not initialized"
        return True, "ok"
    except Exception as e:
        return False, str(e)


@router.get("/ready", response_model=ReadinessResponse, tags=["health"])
async def ready():
    """
    Readiness probe — checks all critical dependencies.
    Returns 200 when ready, 503 when any check fails.
    """
    from fastapi import HTTPException

    checks: dict[str, str] = {}
    all_ok = True

    # Database
    ok, msg = await _check_database()
    checks["database"] = msg
    if not ok:
        all_ok = False

    # Redis (optional — caching/Celery only, not required for core chat)
    ok, msg = _check_redis()
    checks["redis"] = msg if ok else "unavailable (non-critical)"
    # Redis failure is non-fatal — core chat still works without it

    # Vector store
    ok, msg = await _check_vectorstore()
    checks["vectorstore"] = msg
    if not ok:
        all_ok = False

    if all_ok:
        return ReadinessResponse(status="ready", checks=checks)
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ReadinessResponse(
                status="not_ready",
                checks=checks,
                message="One or more dependencies unavailable",
            ).model_dump(),
        )
