"""
Admin User Management API.

Endpoints:
- GET  /api/admin/users              — List all users (paginated, filterable)
- POST /api/admin/users/{user_id}/block   — Block a user
- POST /api/admin/users/{user_id}/unblock — Unblock a user
- PATCH /api/admin/users/{user_id}/role   — Update user role
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import AdminUser, get_current_user, require_admin
from app.models.user import User
from app.models.user_block_log import UserBlockLog

router = APIRouter(prefix="/admin/users", tags=["admin-users"])


# ─────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────

class UserPublic(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_blocked: bool
    blocked_at: Optional[datetime]
    is_deleted: bool
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = {"from_attributes": True}


class PaginatedUsersResponse(BaseModel):
    users: list[UserPublic]
    total: int
    page: int
    page_size: int
    pages: int


class BlockRequest(BaseModel):
    reason: Optional[str] = None


class BlockResponse(BaseModel):
    user_id: str
    is_blocked: bool
    blocked_at: datetime


class UnblockResponse(BaseModel):
    user_id: str
    is_blocked: bool


class UpdateRoleRequest(BaseModel):
    role: str  # "admin" or "user"


class UpdateRoleResponse(BaseModel):
    user_id: str
    role: str
    updated_at: datetime


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedUsersResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name or email"),
    role: Optional[str] = Query(None, description="Filter by role (admin/user)"),
    is_blocked: Optional[bool] = Query(None, description="Filter by blocked status"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    List all users with pagination and filtering.
    Admin access only.
    """
    # Base query
    query = select(User).where(User.is_deleted == False)  # noqa: E712
    count_query = select(func.count(User.id)).where(User.is_deleted == False)  # noqa: E712

    # Apply filters
    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (User.name.ilike(search_filter)) | (User.email.ilike(search_filter))
        )
        count_query = count_query.where(
            (User.name.ilike(search_filter)) | (User.email.ilike(search_filter))
        )

    if role:
        query = query.where(User.role == role)
        count_query = count_query.where(User.role == role)

    if is_blocked is not None:
        query = query.where(User.is_blocked == is_blocked)
        count_query = count_query.where(User.is_blocked == is_blocked)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(User.created_at.desc())

    result = await db.execute(query)
    users = result.scalars().all()

    pages = (total + page_size - 1) // page_size if total > 0 else 1

    return PaginatedUsersResponse(
        users=[UserPublic.model_validate(u) for u in users],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.post("/{user_id}/block", response_model=BlockResponse)
async def block_user(
    user_id: str,
    payload: BlockRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Block a user account.
    Admin access only. Admin cannot block themselves.
    """
    # Prevent admin from blocking themselves
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot block your own account",
        )

    # Find target user
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)  # noqa: E712
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Already blocked?
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already blocked",
        )

    # Block the user
    user.is_blocked = True
    user.blocked_at = datetime.utcnow()

    # Log the action
    block_log = UserBlockLog(
        user_id=user_id,
        action="blocked",
        performed_by=current_user.id,
        reason=payload.reason,
    )
    db.add(block_log)

    await db.commit()
    await db.refresh(user)

    return BlockResponse(
        user_id=user.id,
        is_blocked=user.is_blocked,
        blocked_at=user.blocked_at,
    )


@router.post("/{user_id}/unblock", response_model=UnblockResponse)
async def unblock_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Unblock a user account.
    Admin access only.
    """
    # Find target user
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)  # noqa: E712
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Not blocked?
    if not user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not blocked",
        )

    # Unblock the user
    user.is_blocked = False
    user.blocked_at = None

    # Log the action
    block_log = UserBlockLog(
        user_id=user_id,
        action="unblocked",
        performed_by=current_user.id,
    )
    db.add(block_log)

    await db.commit()
    await db.refresh(user)

    return UnblockResponse(
        user_id=user.id,
        is_blocked=user.is_blocked,
    )


@router.patch("/{user_id}/role", response_model=UpdateRoleResponse)
async def update_user_role(
    user_id: str,
    payload: UpdateRoleRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Update a user's role.
    Admin access only. Admin cannot demote themselves.
    Last admin cannot be demoted.
    """
    # Validate role
    if payload.role not in ("admin", "user"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'admin' or 'user'",
        )

    # Prevent admin from changing their own role
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    # Find target user
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)  # noqa: E712
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Check if this is the last admin
    if user.role == "admin" and payload.role == "user":
        # Count remaining admins
        admin_count_result = await db.execute(
            select(func.count(User.id)).where(
                User.role == "admin",
                User.is_deleted == False,  # noqa: E712
                User.is_blocked == False,  # noqa: E712
            )
        )
        admin_count = admin_count_result.scalar() or 0
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the last admin. At least one admin is required.",
            )

    # Update role
    user.role = payload.role
    user.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(user)

    return UpdateRoleResponse(
        user_id=user.id,
        role=user.role,
        updated_at=user.updated_at,
    )
