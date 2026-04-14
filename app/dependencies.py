"""
FastAPI dependencies — Stage 7 (Governance).

Provides authenticated request context:
- get_current_user        : returns the authenticated User or raises 401
- require_admin           : returns the authenticated User only if role=='admin', else raises 403
- require_admin_or_self   : allows access if user is admin OR user_id matches the authenticated user
- get_audit_context       : extracts IP address and User-Agent from the request for audit logging
"""
import logging
from typing import Annotated, Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db, AsyncSessionLocal
from app.models.user import User

logger = logging.getLogger(__name__)

# auto_error=False so we can support optional/dev-bypass auth flows.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ─────────────────────────────────────────────────────────
# Token decoding
# ─────────────────────────────────────────────────────────

def _decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises HTTPException 401 on invalid/expired tokens.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────────────────
# Database user lookup
# ─────────────────────────────────────────────────────────

async def _get_user_by_id(user_id: str) -> Optional[User]:
    """Fetch a user by ID, returning None if not found or soft-deleted."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(
                User.id == user_id,
                User.is_deleted == False,  # noqa: E712
            )
        )
        return result.scalar_one_or_none()


async def _get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(
        select(User).where(
            User.email == email,
            User.is_deleted == False,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


def _is_dev_bypass_enabled() -> bool:
    return settings.dev_auth_bypass and settings.app_env != "production"


async def _ensure_dev_user(db: AsyncSession) -> User:
    user = await _get_user_by_email(db, settings.dev_auth_email)
    if user:
        return user

    user = User(
        email=settings.dev_auth_email,
        name=settings.dev_auth_name,
        password_hash="dev-bypass",
        role=settings.dev_auth_role,
        is_deleted=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ─────────────────────────────────────────────────────────
# Core dependencies
# ─────────────────────────────────────────────────────────

async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Validates the JWT bearer token and returns the authenticated User.
    Raises 401 if token is invalid, expired, or user does not exist.
    """
    if not token:
        if _is_dev_bypass_enabled():
            return await _ensure_dev_user(db)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = _decode_token(token)
    except HTTPException:
        if _is_dev_bypass_enabled():
            return await _ensure_dev_user(db)
        raise

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing 'sub' claim",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await _get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is blocked
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been blocked. Contact an administrator.",
        )

    return user


async def get_current_user_optional(
    db: Annotated[AsyncSession, Depends(get_db)],
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> Optional[User]:
    """
    Returns the authenticated User if a valid token is provided,
    otherwise returns None. Does NOT raise 401.
    """
    if not token:
        if _is_dev_bypass_enabled():
            return await _ensure_dev_user(db)
        return None
    try:
        payload = _decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        return await _get_user_by_id(user_id)
    except HTTPException:
        if _is_dev_bypass_enabled():
            return await _ensure_dev_user(db)
        return None


def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    Ensures the authenticated user has role='admin'.
    Raises 403 if not.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


def require_admin_or_self(
    target_user_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    Allows access if the current user is an admin OR if the
    target_user_id matches the authenticated user's ID.
    Raises 403 otherwise.
    """
    if current_user.role == "admin" or current_user.id == target_user_id:
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have permission to access this resource",
    )


# ─────────────────────────────────────────────────────────
# Audit context helper
# ─────────────────────────────────────────────────────────

def get_audit_context(request: Request) -> tuple[Optional[str], Optional[str]]:
    """
    Extract IP address and User-Agent from the incoming request.

    Returns:
        Tuple of (ip_address, user_agent)
    """
    ip_address = (
        request.client.host
        if request.client
        else None
    )
    # Support X-Forwarded-For in proxied setups
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip_address = forwarded.split(",")[0].strip()

    user_agent = request.headers.get("user-agent")
    return ip_address, user_agent


# ─────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────

CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
