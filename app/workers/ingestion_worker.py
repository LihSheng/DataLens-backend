"""Celery ingestion worker tasks.

Tasks:
- ingestion.process_document  — run full ingestion pipeline
- ingestion.reindex_document   — re-chunk and re-index an existing document
- ingestion.delete_document    — soft-delete (mark is_active_version=False)
"""
import logging
import asyncio
from typing import Dict, Any, Optional

from celery import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.workers.celery_app import celery_app
from app.config import settings

# Import all models to ensure FK relationships are resolved by SQLAlchemy
from app.models import user as _user_model  # noqa: F401
from app.models import document as _document_model  # noqa: F401
from app.models import conversation as _conversation_model  # noqa: F401

logger = logging.getLogger(__name__)

# ── Async DB session factory (for Celery workers) ─────────────────────────────

_engine = None
_SessionFactory = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


def _get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = async_sessionmaker(
            _get_engine(), expire_on_commit=False
        )
    return _SessionFactory


async def _run_async(coro):
    """Run an async coroutine from within a Celery synchronous task."""
    return asyncio.run(coro)


# ── Helpers ────────────────────────────────────────────────────────────────────


async def _get_db_session() -> AsyncSession:
    factory = _get_session_factory()
    return factory()


# ── Celery Tasks ───────────────────────────────────────────────────────────────


@celery_app.task(
    name="ingestion.process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def process_document(self: Task, file_path: str, user_id: str, document_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline on an uploaded document.

    This task runs synchronously in a Celery worker.
    The async pipeline is executed via asyncio.run().
    """
    from app.ingestion.pipeline import run_ingestion_pipeline

    logger.info(
        f"[Celery] process_document: doc={document_id}, file={file_path}, user={user_id}"
    )

    try:
        # Import here to avoid top-level circular deps
        options = options or {}

        # Run async pipeline synchronously inside Celery
        async def _run():
            async with _get_session_factory()() as db:
                result = await run_ingestion_pipeline(
                    file_path=file_path,
                    user_id=user_id,
                    document_id=document_id,
                    options=options,
                    db=db,
                )
                return result

        result = asyncio.run(_run())
        logger.info(f"[Celery] process_document complete: {document_id} -> {result['status']}")
        return result

    except Exception as exc:
        logger.exception(f"[Celery] process_document failed for {document_id}: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc)


@celery_app.task(
    name="ingestion.reindex_document",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def reindex_document(self: Task, document_id: str) -> Dict[str, Any]:
    """
    Re-read the original file, re-chunk, re-index, and increment version.

    - Fetches the current Document record
    - Creates a new version (increments version, sets is_active_version=False on old)
    - Re-runs chunking and vectorstore indexing
    - Updates status to "ready"
    """
    from app.ingestion.pipeline import run_ingestion_pipeline
    from app.ingestion.parsers import parse_document
    from app.ingestion.chunker import chunk_document
    from app.ingestion.pii import detect_pii, entities_to_json
    from app.ingestion.ocr import ocr_pdf
    from app.services.vectorstore_service import get_vectorstore, add_documents_with_bm25
    from app.models.document import Document
    from pathlib import Path

    logger.info(f"[Celery] reindex_document: doc={document_id}")

    async def _run():
        db = await _get_db_session()
        try:
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                return {"error": f"Document {document_id} not found"}

            # Deactivate current version
            doc.is_active_version = False
            await db.flush()

            # Create new version
            new_doc = Document(
                user_id=doc.user_id,
                name=doc.name,
                file_path=doc.file_path,
                size=doc.size,
                extension=doc.extension,
                status="processing",
                version=doc.version + 1,
                parent_document_id=doc.id,
                is_active_version=True,
                chunking_strategy=doc.chunking_strategy,
            )
            db.add(new_doc)
            await db.flush()

            # Re-parse
            parse_result = parse_document(doc.file_path)
            text = parse_result["text"]
            needs_ocr = parse_result["needs_ocr"]
            ocr_used = False

            if needs_ocr:
                suffix = Path(doc.file_path).suffix.lower()
                if suffix == ".pdf":
                    ocr_results = ocr_pdf(doc.file_path)
                    if ocr_results:
                        text = "\n\n".join(r["text"] for r in ocr_results)
                        ocr_used = True

            # PII detection (no redaction on reindex)
            pii_entities = detect_pii(text, use_presidio=False)
            pii_json = entities_to_json(pii_entities)

            # Chunk
            doc_metadata = {
                "source": doc.file_path,
                "filename": doc.name,
                "extension": doc.extension,
                "document_id": new_doc.id,
                "user_id": doc.user_id,
            }
            chunks = chunk_document(
                text=text,
                metadata=doc_metadata,
                strategy=doc.chunking_strategy or "recursive",
            )

            # Index
            vs = get_vectorstore()
            add_documents_with_bm25(chunks)

            # Update new version
            new_doc.status = "ready"
            new_doc.ocr_applied = ocr_used
            new_doc.pii_entities_found = pii_json

            await db.commit()

            logger.info(
                f"[Celery] reindex complete: {document_id} -> new version {new_doc.version}"
            )
            return {
                "document_id": new_doc.id,
                "new_version": new_doc.version,
                "chunks_ingested": len(chunks),
                "status": "ready",
            }
        except Exception as exc:
            logger.exception(f"[Celery] reindex_document failed: {exc}")
            await db.rollback()
            raise

    return asyncio.run(_run())


@celery_app.task(
    name="ingestion.delete_document",
    bind=True,
    acks_late=True,
)
def delete_document(self: Task, document_id: str) -> Dict[str, Any]:
    """
    Soft-delete a document: mark the active version as inactive.
    Also marks all child versions as inactive.
    """
    from app.models.document import Document

    logger.info(f"[Celery] delete_document: doc={document_id}")

    async def _run():
        db = await _get_db_session()
        try:
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            doc = result.scalar_one_or_none()
            if doc is None:
                return {"error": f"Document {document_id} not found"}

            doc.is_active_version = False
            doc.status = "deleted"

            # Also deactivate all child versions
            await db.execute(
                select(Document).where(Document.parent_document_id == document_id)
            )
            children_result = await db.execute(
                select(Document).where(Document.parent_document_id == document_id)
            )
            for child in children_result.scalars().all():
                child.is_active_version = False
                child.status = "deleted"

            await db.commit()
            logger.info(f"[Celery] delete_document complete: {document_id}")
            return {"document_id": document_id, "status": "deleted"}

        except Exception as exc:
            logger.exception(f"[Celery] delete_document failed: {exc}")
            await db.rollback()
            raise

    return asyncio.run(_run())
