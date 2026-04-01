"""
Shared FastAPI dependencies.
Auth stubs — will be wired up in later stages.
"""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """
    Stub: validates JWT token and returns the current user.
    Full implementation lands in Stage 7 (Governance).
    """
    # TODO(Stage7): real JWT decode + db lookup
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """
    Stub: returns user if token is valid, otherwise None.
    """
    # TODO(Stage7): real implementation
    return None


def require_admin(current_user=Depends(get_current_user)):
    """
    Stub: raises 403 if current user is not admin.
    """
    # TODO(Stage7): real admin check
    raise HTTPException(status_code=403, detail="Admin access required")
