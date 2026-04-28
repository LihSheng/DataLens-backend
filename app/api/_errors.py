"""
Standardized error schema and FastAPI exception handlers.

All errors follow: { error: { code: str, message: str, details: Any?, request_id: str } }
Correlation ID is generated per-request and included in all responses.
"""
import logging
import uuid
from typing import Any, Optional

from fastapi import Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ─── Error Response Model ───────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None
    request_id: str = ""


def build_error(
    code: str,
    message: str,
    details: Optional[Any] = None,
    request_id: str = "",
) -> dict:
    return {"error": ErrorResponse(code=code, message=message, details=details, request_id=request_id).model_dump()}


# ─── Request ID middleware ──────────────────────────────────────────────────

REQUEST_ID_CTX_KEY = "request_id"


async def request_id_middleware(request: Request, call_next):
    """Generate a correlation ID for every request, store in request.state."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# ─── Exception handlers ───────────────────────────────────────────────────

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException.detail (string or dict) to standard error shape."""
    request_id = getattr(request.state, REQUEST_ID_CTX_KEY, "")

    # Already formatted as our error dict
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    # String detail — wrap in standard shape
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            request_id=request_id,
        ),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, REQUEST_ID_CTX_KEY, "")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=build_error(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=exc.errors(),
            request_id=request_id,
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any unhandled exception — always returns JSON, never plaintext."""
    request_id = getattr(request.state, REQUEST_ID_CTX_KEY, str(uuid.uuid4()))
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=build_error(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred. Please try again.",
            details=None,
            request_id=request_id,
        ),
    )
