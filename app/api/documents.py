"""
Document ingestion API — Sprint 2.

Endpoints:
- POST /upload          — async upload + Celery pipeline
- GET  /{document_id}/status
- POST /{document_id}/reindex
- DELETE /{document_id}
- POST /ingest           — backward-compatible sync endpoint (dev/testing)
"""
import os
import uuid
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.document import Document
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ensure_upload_dir() -> Path:
    path = Path(settings.upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_upload_file(file: UploadFile) -> tuple[str, int]:
    """
    Save an UploadFile to the upload directory.

    Returns:
        (absolute_file_path, file_size)
    """
    upload_dir = _ensure_upload_dir()
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    tmp_path = upload_dir / f"{uuid.uuid4()}{ext}"

    file_size = 0
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        f.flush()
        f.seek(0)
        file_size = f.tell()

    return str(tmp_path), file_size


async def _get_document_or_404(
    document_id: str,
    db: AsyncSession,
    user: User,
) -> Document:
    """Fetch a document by ID, verifying ownership. Raises 404 if not found."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    redact_pii: bool = False,
):
    """
    Upload a document and queue it for async ingestion via Celery.

    Supported formats: .pdf, .docx, .html, .csv, .txt, .md, .xlsx, .xls

    Returns immediately with document_id and "processing" status.
    Poll GET /{document_id}/status for completion.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".html", ".csv", ".txt", ".md", ".xlsx", ".xls"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed)}",
        )

    # Enforce max upload size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Save file to disk
    try:
        file_path, file_size = _save_upload_file(file)
    except Exception as e:
        logger.exception(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    if file_size > max_bytes:
        os.unlink(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB",
        )

    # Create Document DB record with status="processing"
    doc = Document(
        user_id=current_user.id,
        name=file.filename,
        file_path=file_path,
        size=file_size,
        extension=suffix,
        status="processing",
        chunking_strategy=chunk_strategy,
        version=1,
        is_active_version=True,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Dispatch Celery task
    try:
        from app.workers.ingestion_worker import process_document

        process_document.delay(
            file_path=file_path,
            user_id=current_user.id,
            document_id=doc.id,
            options={
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "redact_pii": redact_pii,
                "use_presidio": False,
                "enable_semantic": False,
            },
        )
        logger.info(f"[API] Queued document {doc.id} for processing")
    except Exception as e:
        logger.exception(f"[API] Failed to dispatch Celery task: {e}")
        # Document is still created — user can retry or reindex later
        pass

    return {
        "document_id": doc.id,
        "name": doc.name,
        "status": "processing",
        "message": "Document queued for processing",
        "version": doc.version,
    }


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the current status and metadata of a document.

    Includes: status, parse_error, version, ocr_applied, pii_entities_found,
    chunking_strategy, is_active_version, created_at.
    """
    doc = await _get_document_or_404(document_id, db, current_user)

    return {
        "document_id": doc.id,
        "name": doc.name,
        "status": doc.status,
        "parse_error": doc.parse_error,
        "version": doc.version,
        "ocr_applied": doc.ocr_applied,
        "pii_entities_found": doc.pii_entities_found,
        "chunking_strategy": doc.chunking_strategy,
        "is_active_version": doc.is_active_version,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "extension": doc.extension,
        "size": doc.size,
    }


@router.post("/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Queue a document for reindexing.

    Creates a new version (increments version number), re-parses the original
    file, re-chunks, and re-indexes. The old version is marked inactive.
    """
    doc = await _get_document_or_404(document_id, db, current_user)

    # Check the file still exists
    if not os.path.exists(doc.file_path):
        raise HTTPException(
            status_code=404,
            detail="Original file no longer found on disk. Cannot reindex.",
        )

    try:
        from app.workers.ingestion_worker import reindex_document

        result = reindex_document.delay(document_id)
        # Note: result is a Celery AsyncResult — we return immediately
        logger.info(f"[API] Queued reindex for document {document_id}")
    except Exception as e:
        logger.exception(f"[API] Failed to dispatch reindex task: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue reindex task")

    return {
        "message": "Reindex queued",
        "document_id": document_id,
        "current_version": doc.version,
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Soft-delete a document and all its versions.

    Sets is_active_version=False on the document and all child versions.
    The document is no longer served in search results.
    """
    doc = await _get_document_or_404(document_id, db, current_user)

    try:
        from app.workers.ingestion_worker import delete_document

        delete_document.delay(document_id)
        logger.info(f"[API] Queued delete for document {document_id}")
    except Exception as e:
        logger.exception(f"[API] Failed to dispatch delete task: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue delete task")

    return {"message": "Delete queued", "document_id": document_id}


# ── Backward-compatible sync endpoint (dev/testing) ───────────────────────────


@router.post("/ingest")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    redact_pii: bool = False,
):
    """
    Synchronous ingestion endpoint (dev/testing only).

    Processes files inline and returns results immediately.
    For production use, prefer POST /upload for async processing.

    Supported formats: .pdf, .docx, .html, .csv, .txt, .md, .xlsx, .xls
    """
    from app.ingestion.parsers import parse_document
    from app.ingestion.ocr import ocr_pdf
    from app.ingestion.pii import detect_pii, redact_pii as redact_fn, entities_to_json
    from app.ingestion.chunker import chunk_document
    from app.services.vectorstore_service import get_vectorstore, add_documents_with_bm25

    total_chunks = 0
    total_pii = 0
    processed = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        allowed = {".pdf", ".docx", ".html", ".csv", ".txt", ".md", ".xlsx", ".xls"}
        if suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}",
            )

        # Save to temp file
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Parse
            result = parse_document(tmp_path)
            text = result["text"]
            needs_ocr = result["needs_ocr"]

            if needs_ocr and suffix == ".pdf":
                ocr_results = ocr_pdf(tmp_path)
                if ocr_results:
                    text = "\n\n".join(r["text"] for r in ocr_results)

            # PII
            pii_entities = detect_pii(text, use_presidio=False)
            total_pii += len(pii_entities)

            if redact_pii and pii_entities:
                text = redact_fn(text, pii_entities)

            # Chunk
            metadata = {
                "source": tmp_path,
                "filename": file.filename,
                "extension": suffix,
            }
            chunks = chunk_document(
                text=text,
                metadata=metadata,
                strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Index
            vs = get_vectorstore()
            add_documents_with_bm25(chunks)

            total_chunks += len(chunks)
            processed.append(file.filename)

        finally:
            os.unlink(tmp_path)

    return {
        "message": f"Ingested {total_chunks} chunks from {len(processed)} file(s)",
        "file_count": len(processed),
        "chunks": total_chunks,
        "pii_found": total_pii,
        "files": processed,
    }
