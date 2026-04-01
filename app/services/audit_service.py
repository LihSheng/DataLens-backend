"""
Audit service — Stage 7 (Governance).

Provides a thin, async-safe wrapper around AuditLog model writes.
Every admin action should call `audit()` to record what happened.
"""
import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.models.user import User

logger = logging.getLogger(__name__)


async def audit(
    db: AsyncSession,
    user: User,
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> AuditLog:
    """
    Write a single immutable audit log entry.

    Args:
        db: Async database session
        user: The authenticated user performing the action
        action: Short action name, e.g. 'create_user', 'delete_experiment'
        resource: Resource type, e.g. 'user', 'experiment', 'settings'
        resource_id: UUID of the affected resource (if applicable)
        details: Arbitrary JSON-serialisable context
        ip_address: Client IP address
        user_agent: Client User-Agent string

    Returns:
        The created AuditLog row (not committed — caller is responsible)
    """
    entry = AuditLog(
        user_id=user.id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        created_at=datetime.utcnow(),
    )
    db.add(entry)
    # Do NOT commit here — caller controls the transaction boundary
    logger.info(
        f"AUDIT [{user.email}] {action} on {resource} "
        f"(id={resource_id}) from {ip_address}"
    )
    return entry


async def audit_from_context(
    db: AsyncSession,
    user_id: str,
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> AuditLog:
    """
    Write an audit log entry using only a user_id (no full User object).
    Used when the full User object isn't available in the call chain.
    """
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        created_at=datetime.utcnow(),
    )
    db.add(entry)
    logger.info(
        f"AUDIT [user={user_id}] {action} on {resource} "
        f"(id={resource_id}) from {ip_address}"
    )
    return entry
