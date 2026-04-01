"""
Export API — download conversations as Markdown or PDF.

Endpoints:
- GET  /api/exports/conversation/{conversation_id}/markdown
- GET  /api/exports/conversation/{conversation_id}/pdf
- POST /api/exports/conversation/{conversation_id}/pdf  (async → job id)
"""
import logging
from datetime import datetime
from typing import Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.export import conversation_to_markdown, markdown_to_pdf_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/exports", tags=["exports"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class PDFJobStatus(BaseModel):
    job_id: str
    status: str  # "queued" | "completed" | "failed"
    conversation_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class PDFJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# In-memory job store (replace with Redis/Celery in production)
_pdf_jobs: dict[str, dict] = {}


async def _get_conversation_with_messages(
    conversation_id: str,
    user_id: str,
    db: AsyncSession,
) -> tuple[Conversation, list[Message]]:
    """Load conversation + messages, verifying ownership."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == user_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    # Sort messages chronologically
    messages = sorted(conv.messages, key=lambda m: m.created_at)
    return conv, messages


def _slug(title: str) -> str:
    """Create a URL-safe slug from a title."""
    import re
    slug = re.sub(r"[^\w\s-]", "", title).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug or "conversation"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/conversation/{conversation_id}/markdown",
    summary="Export conversation as Markdown",
)
async def export_markdown(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Download a conversation as a `.md` file.

    Returns a streamed Markdown file with the conversation title and all
    messages rendered in a readable format.
    """
    conv, messages = await _get_conversation_with_messages(
        conversation_id, current_user.id, db
    )

    md_text = conversation_to_markdown(conv, messages)
    filename = f"{_slug(conv.title)}_{conversation_id[:8]}.md"

    return StreamingResponse(
        iter([md_text]),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
        },
    )


@router.get(
    "/conversation/{conversation_id}/pdf",
    summary="Export conversation as PDF",
)
async def export_pdf(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Synchronously convert a conversation to PDF and return it as a download.

    For large conversations prefer the async POST endpoint.
    """
    conv, messages = await _get_conversation_with_messages(
        conversation_id, current_user.id, db
    )

    md_text = conversation_to_markdown(conv, messages)

    try:
        pdf_bytes = await markdown_to_pdf_bytes_async(md_text)
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )

    filename = f"{_slug(conv.title)}_{conversation_id[:8]}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "Content-Length": str(len(pdf_bytes)),
        },
    )


@router.post(
    "/conversation/{conversation_id}/pdf",
    response_model=PDFJobResponse,
    summary="Generate PDF async",
)
async def generate_pdf_async(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Queue a PDF generation job and return a job ID.

    The actual PDF generation runs in a thread pool. Poll
    GET /api/exports/jobs/{job_id} for completion status.
    """
    # Validate conversation exists (and user owns it)
    await _get_conversation_with_messages(conversation_id, current_user.id, db)

    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    _pdf_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "conversation_id": conversation_id,
        "created_at": now,
        "completed_at": None,
        "error": None,
    }

    # Kick off generation in background (fire-and-forget)
    import asyncio
    asyncio.create_task(_run_pdf_job(job_id, conversation_id, current_user.id, db))

    return PDFJobResponse(
        job_id=job_id,
        status="queued",
        message="PDF generation job queued",
    )


async def _run_pdf_job(job_id: str, conversation_id: str, user_id: str, db: AsyncSession):
    """Background task that generates the PDF and updates the job record."""
    from app.db.session import AsyncSessionLocal

    job = _pdf_jobs.get(job_id)
    if not job:
        return

    try:
        async with AsyncSessionLocal() as s:
            conv, messages = await _get_conversation_with_messages(conversation_id, user_id, s)
            md_text = conversation_to_markdown(conv, messages)
            pdf_bytes = await markdown_to_pdf_bytes_async(md_text)

        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow()
        job["pdf_bytes"] = pdf_bytes
    except Exception as exc:
        logger.exception(f"[export] PDF job {job_id} failed")
        job["status"] = "failed"
        job["completed_at"] = datetime.utcnow()
        job["error"] = str(exc)


@router.get("/jobs/{job_id}/status", response_model=PDFJobStatus)
async def get_pdf_job_status(job_id: str):
    """Poll the status of an async PDF generation job."""
    job = _pdf_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return PDFJobStatus(**job)


@router.get("/jobs/{job_id}/download")
async def download_pdf_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Download the completed PDF for a job."""
    job = _pdf_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not ready: {job['status']}",
        )

    pdf_bytes = job.get("pdf_bytes")
    if not pdf_bytes:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="PDF bytes not available")

    filename = f"conversation_{job['conversation_id'][:8]}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
            "Content-Length": str(len(pdf_bytes)),
        },
    )
