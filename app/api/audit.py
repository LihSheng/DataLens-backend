"""
Audit API — Stage 7 (Governance).

POST /api/admin/audit         — list audit logs (admin only), with filters + pagination
GET  /api/admin/audit/export  — export audit logs as CSV (admin only)

All endpoints require admin role (enforced via require_admin dependency).
"""
import csv
import io
import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import AdminUser, get_audit_context
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/audit", tags=["audit"])


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class AuditLogResponse(BaseModel):
    id: str
    user_id: str
    action: str
    resource: str
    resource_id: Optional[str]
    details: Optional[dict]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int
    limit: int
    offset: int


class AuditLogFilter(BaseModel):
    """Query filters for the audit log list endpoint."""
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _entry_to_response(entry: AuditLog) -> AuditLogResponse:
    return AuditLogResponse(
        id=entry.id,
        user_id=entry.user_id,
        action=entry.action,
        resource=entry.resource,
        resource_id=entry.resource_id,
        details=entry.details,
        ip_address=entry.ip_address,
        user_agent=entry.user_agent,
        created_at=entry.created_at.isoformat() if entry.created_at else "",
    )


def _apply_filters(stmt, filters: AuditLogFilter):
    """Apply optional filters to a SQLAlchemy select statement."""
    if filters.user_id:
        stmt = stmt.where(AuditLog.user_id == filters.user_id)
    if filters.action:
        stmt = stmt.where(AuditLog.action == filters.action)
    if filters.resource:
        stmt = stmt.where(AuditLog.resource == filters.resource)
    if filters.resource_id:
        stmt = stmt.where(AuditLog.resource_id == filters.resource_id)
    if filters.date_from:
        stmt = stmt.where(AuditLog.created_at >= filters.date_from)
    if filters.date_to:
        stmt = stmt.where(AuditLog.created_at <= filters.date_to)
    return stmt


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("", response_model=AuditLogListResponse)
async def list_audit_logs(
    request: Request,
    # Filter params
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action: Optional[str] = Query(None, description="Filter by action name"),
    resource: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    date_from: Optional[datetime] = Query(None, description="Filter from date (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="Filter to date (ISO 8601)"),
    # Pagination
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    # Auth
    current_user: AdminUser = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List audit log entries with optional filters and pagination.

    Returns newest entries first. All filters are AND-combined.
    Only accessible by admin users.
    """
    filters = AuditLogFilter(
        user_id=user_id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        date_from=date_from,
        date_to=date_to,
    )

    # Count query
    count_stmt = select(func.count(AuditLog.id))
    count_stmt = _apply_filters(count_stmt, filters)
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Data query
    data_stmt = (
        select(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    data_stmt = _apply_filters(data_stmt, filters)
    result = await db.execute(data_stmt)
    rows = result.scalars().all()

    # Audit the read action itself
    ip_address, user_agent = get_audit_context(request)
    from app.services.audit_service import audit
    await audit(
        db=db,
        user=current_user,
        action="list_audit_logs",
        resource="audit_log",
        resource_id=None,
        details={
            "filters": {
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "resource_id": resource_id,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
            },
            "limit": limit,
            "offset": offset,
        },
        ip_address=ip_address,
        user_agent=user_agent,
    )
    await db.commit()

    return AuditLogListResponse(
        items=[_entry_to_response(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/export")
async def export_audit_logs(
    request: Request,
    # Same filters as list endpoint
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    # Auth
    current_user: AdminUser = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Export audit logs as a CSV file.

    Returns all matching rows (up to a reasonable limit of 10 000)
    as a streamed CSV download.
    Only accessible by admin users.
    """
    filters = AuditLogFilter(
        user_id=user_id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        date_from=date_from,
        date_to=date_to,
    )

    stmt = (
        select(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(10_000)
    )
    stmt = _apply_filters(stmt, filters)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    # Audit the export action
    ip_address, user_agent = get_audit_context(request)
    from app.services.audit_service import audit
    await audit(
        db=db,
        user=current_user,
        action="export_audit_logs",
        resource="audit_log",
        resource_id=None,
        details={"row_count": len(rows)},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    await db.commit()

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "user_id", "action", "resource", "resource_id",
        "details", "ip_address", "user_agent", "created_at",
    ])
    for r in rows:
        writer.writerow([
            r.id,
            r.user_id,
            r.action,
            r.resource,
            r.resource_id or "",
            json.dumps(r.details) if r.details else "",
            r.ip_address or "",
            r.user_agent or "",
            r.created_at.isoformat() if r.created_at else "",
        ])

    output.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_logs_{timestamp}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )
