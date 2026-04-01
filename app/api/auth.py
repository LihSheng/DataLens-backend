"""
Auth API for frontend compatibility.

Endpoints:
- POST /api/auth/login
- POST /api/auth/logout
- GET  /api/me
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.dependencies import get_current_user
from app.models.user import User

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginUser(BaseModel):
    id: str
    email: str
    name: str
    role: str


class LoginResponse(BaseModel):
    user: LoginUser
    accessToken: str


def _issue_token(user: User) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _is_dev_login_allowed() -> bool:
    return settings.dev_auth_bypass and settings.app_env != "production"


@router.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Dev-friendly login endpoint.

    In MVP mode this allows credential login when DEV_AUTH_BYPASS is enabled.
    """
    if not _is_dev_login_allowed():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential auth is not enabled in this environment.",
        )

    result = await db.execute(
        select(User).where(
            User.email == payload.email,
            User.is_deleted == False,  # noqa: E712
        )
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = User(
            email=payload.email,
            name=payload.email.split("@")[0] or "User",
            password_hash="dev-bypass",
            role=settings.dev_auth_role,
            is_deleted=False,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    token = _issue_token(user)
    return LoginResponse(
        user=LoginUser(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
        ),
        accessToken=token,
    )


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout():
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=LoginUser)
async def me(current_user: User = Depends(get_current_user)):
    return LoginUser(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        role=current_user.role,
    )
