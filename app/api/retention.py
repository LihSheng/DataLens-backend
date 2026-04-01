"""
Retention Policy API — Stage 7 (Governance).

GET  /api/settings/retention      — get current retention policy
PATCH /api/settings/retention      — update retention policy (admin only)

Retention policies control automatic data lifecycle cleanup.
A Celery beat task (`retention_task`) runs daily to enforce them.
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import AdminUser, get_audit_context
from app.models.retention_policy import RetentionPolicy
from app.services.audit_service import audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings/retention", tags=["retention"])


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class RetentionPolicyEntry(BaseModel):
    resource: str
    retention_days: int = Field(..., ge=1, description="Retention period in days")
    is_active: bool = True


class RetentionPolicyResponse(BaseModel):
    policies: list[RetentionPolicyEntry]
    updated_at: Optional[str] = None


class RetentionPolicyUpdateRequest(BaseModel):
    resource: str = Field(..., description="Resource type: 'conversations', 'messages', or 'feedback'")
    retention_days: int = Field(..., ge=1, le=3650, description="Retention period in days (1–3650)")
    is_active: bool = Field(default=True)


class RetentionPolicyUpdateResponse(BaseModel):
    resource: str
    retention_days: int
    is_active: bool
    updated_at: str


# ─────────────────────────────────────────────────────────
# Defaults — inserted on first GET if the table is empty
# ─────────────────────────────────────────────────────────

DEFAULT_POLICIES = [
    {"resource": "conversations", "retention_days": 90},
    {"resource": "messages", "retention_days": 90},
    {"resource": "feedback", "retention_days": 90},
]


async def _ensure_defaults(db: AsyncSession) -> list[RetentionPolicy]:
    """
    If the retention_policy table is empty, seed it with defaults.
    Returns all active policies.
    """
    result = await db.execute(select(RetentionPolicy))
    existing = result.scalars().all()

    if not existing:
        rows = [
            RetentionPolicy(
                resource=p["resource"],
                retention_days=p["retention_days"],
                is_active=True,
            )
            for p in DEFAULT_POLICIES
        ]
        for r in rows:
            db.add(r)
        await db.commit()
        # Re-fetch after commit
        result = await db.execute(select(RetentionPolicy))
        return list(result.scalars().all())

    return list(existing)


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.get("", response_model=RetentionPolicyResponse)
async def get_retention_policy(
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current retention policies for all resource types.

    Returns the list of policies, creating default entries on first call.
    This endpoint is publicly readable (no auth required).
    """
    policies = await _ensure_defaults(db)

    # Get the most-recently updated timestamp
    updated_at = None
    if policies:
        max_updated = max(r.updated_at for r in policies if r.updated_at)
        updated_at = max_updated.isoformat() if max_updated else None

    return RetentionPolicyResponse(
        policies=[
            RetentionPolicyEntry(
                resource=p.resource,
                retention_days=p.retention_days,
                is_active=p.is_active,
            )
            for p in policies
        ],
        updated_at=updated_at,
    )


@router.patch("", response_model=RetentionPolicyUpdateResponse)
async def update_retention_policy(
    request: Request,
    payload: RetentionPolicyUpdateRequest,
    current_user: AdminUser = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Update a specific retention policy entry.

    Only admins can update retention policies. All changes are audit-logged.
    """
    # Find existing policy for this resource
    result = await db.execute(
        select(RetentionPolicy).where(RetentionPolicy.resource == payload.resource)
    )
    policy = result.scalar_one_or_none()

    ip_address, user_agent = get_audit_context(request)

    if policy is None:
        # Create a new entry
        policy = RetentionPolicy(
            resource=payload.resource,
            retention_days=payload.retention_days,
            is_active=payload.is_active,
            updated_at=datetime.utcnow(),
        )
        db.add(policy)
        action = "create_retention_policy"
    else:
        # Update existing
        policy.retention_days = payload.retention_days
        policy.is_active = payload.is_active
        policy.updated_at = datetime.utcnow()
        action = "update_retention_policy"

    # Audit log
    await audit(
        db=db,
        user=current_user,
        action=action,
        resource="retention_policy",
        resource_id=policy.id if policy.id else None,
        details={
            "resource": payload.resource,
            "retention_days": payload.retention_days,
            "is_active": payload.is_active,
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    await db.commit()
    await db.refresh(policy)

    return RetentionPolicyUpdateResponse(
        resource=policy.resource,
        retention_days=policy.retention_days,
        is_active=policy.is_active,
        updated_at=policy.updated_at.isoformat() if policy.updated_at else "",
    )
