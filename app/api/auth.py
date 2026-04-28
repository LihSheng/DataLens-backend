"""
Auth API for frontend compatibility.

Endpoints:
- POST /api/auth/register  — User registration
- POST /api/auth/login    — Login
- POST /api/auth/logout   — Logout
- GET  /api/me            — Get current user
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.api._errors import build_error

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least 1 uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least 1 digit")
        if not re.search(r"[@$!%*?&]", v):
            raise ValueError("Password must contain at least 1 special character (@$!%*?&)")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2 or len(v) > 100:
            raise ValueError("Name must be between 2 and 100 characters")
        return v


class LoginUser(BaseModel):
    id: str
    email: str
    name: str
    role: str


class LoginResponse(BaseModel):
    user: LoginUser
    accessToken: str


class RegisterResponse(BaseModel):
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


@router.post("/auth/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    User registration endpoint.

    Creates a new user account with role='user' and returns a JWT token.
    """
    # Check if email already exists
    result = await db.execute(
        select(User).where(
            User.email == payload.email,
            User.is_deleted == False,  # noqa: E712
        )
    )
    existing_user = result.scalar_one_or_none()
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=build_error(code="EMAIL_ALREADY_EXISTS", message="Email already registered"),
        )

    # Create new user (password stored as hash in production, dev-bypass for dev mode)
    user = User(
        email=payload.email,
        name=payload.name,
        password_hash=payload.password,  # In production, use proper hashing
        role="user",
        is_deleted=False,
        is_blocked=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = _issue_token(user)
    return RegisterResponse(
        user=LoginUser(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
        ),
        accessToken=token,
    )


@router.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Dev-friendly login endpoint.

    In MVP mode this allows credential login when DEV_AUTH_BYPASS is enabled.
    """
    if not _is_dev_login_allowed():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=build_error(code="AUTH_NOT_ENABLED", message="Credential auth is not enabled in this environment."),
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
            is_blocked=False,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # Check if user is blocked
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=build_error(code="ACCOUNT_BLOCKED", message="Your account has been blocked. Contact an administrator."),
        )

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
