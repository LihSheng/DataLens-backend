"""Main ingestion pipeline.

Coordinates: parsing → OCR fallback → PII detection/redaction → chunking → vector indexing.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.ingestion.parsers import parse_document
from app.ingestion.ocr import ocr_pdf
from app.ingestion.pii import detect_pii, redact_pii, entities_to_json
from app.ingestion.chunker import chunk_document
from app.services.vectorstore_service import get_vectorstore, add_documents_with_bm25

logger = logging.getLogger(__name__)


async def run_ingestion_pipeline(
    file_path: str,
    user_id: str,
    document_id: str,
    options: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline on a document.

    Args:
        file_path:    Absolute path to the uploaded file.
        user_id:      ID of the owning user.
        document_id:  ID of the Document DB record.
        options:      Pipeline options:
                      - chunk_strategy: "recursive" | "fixed" | "semantic"
                      - chunk_size: int
                      - chunk_overlap: int
                      - redact_pii: bool
                      - use_presidio: bool
                      - enable_semantic: bool
        db:           Async SQLAlchemy session.

    Returns:
        {
            "document_id": str,
            "chunks_ingested": int,
            "pii_found": int,
            "ocr_used": bool,
            "status": str,         # "ready" | "failed"
            "parse_error": str | None,
        }
    """
    opts = options or {}
    chunk_strategy = opts.get("chunk_strategy", "recursive")
    chunk_size = opts.get("chunk_size", 1000)
    chunk_overlap = opts.get("chunk_overlap", 200)
    redact_pii_flag = opts.get("redact_pii", False)
    use_presidio = opts.get("use_presidio", False)
    enable_semantic = opts.get("enable_semantic", False)

    # ── 1. Parse document ──────────────────────────────────────────────────────
    logger.info(f"[Pipeline] Parsing {file_path}")
    parse_result = parse_document(file_path)
    text = parse_result["text"]
    pages = parse_result["pages"]
    parse_meta = parse_result["metadata"]
    needs_ocr = parse_result["needs_ocr"]

    ocr_used = False

    # ── 2. OCR fallback ─────────────────────────────────────────────────────────
    if needs_ocr:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf" and needs_ocr:
            logger.info(f"[Pipeline] Running OCR on {file_path}")
            ocr_results = ocr_pdf(file_path)
            if ocr_results:
                ocr_texts = [r["text"] for r in ocr_results]
                text = "\n\n".join(ocr_texts)
                ocr_used = True
                logger.info(f"[Pipeline] OCR extracted {len(text)} chars")

    # If OCR also produced nothing, bail out with an error
    if not text or len(text.strip()) < 50:
        logger.warning(f"[Pipeline] No text extracted from {file_path} (OCR={ocr_used})")
        await _update_document_status(
            db, document_id,
            status="failed",
            parse_error="No text could be extracted from the document",
            ocr_applied=ocr_used,
            pii_entities_found="[]",
        )
        return {
            "document_id": document_id,
            "chunks_ingested": 0,
            "pii_found": 0,
            "ocr_used": ocr_used,
            "status": "failed",
            "parse_error": "No text could be extracted",
        }

    # ── 3. PII detection ───────────────────────────────────────────────────────
    pii_entities = detect_pii(text, use_presidio=use_presidio)
    pii_json = entities_to_json(pii_entities)

    if redact_pii_flag and pii_entities:
        logger.info(f"[Pipeline] Redacting {len(pii_entities)} PII entities")
        text = redact_pii(text, pii_entities)

    # ── 4. Chunking ─────────────────────────────────────────────────────────────
    doc_metadata = {
        **parse_meta,
        "document_id": document_id,
        "user_id": user_id,
    }

    chunks = chunk_document(
        text=text,
        metadata=doc_metadata,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_semantic=enable_semantic,
    )

    if not chunks:
        logger.warning(f"[Pipeline] No chunks produced for {file_path}")
        await _update_document_status(
            db, document_id,
            status="failed",
            parse_error="Chunking produced no output",
            ocr_applied=ocr_used,
            pii_entities_found=pii_json,
        )
        return {
            "document_id": document_id,
            "chunks_ingested": 0,
            "pii_found": len(pii_entities),
            "ocr_used": ocr_used,
            "status": "failed",
            "parse_error": "Chunking produced no output",
        }

    # ── 5. Add to vector store ─────────────────────────────────────────────────
    try:
        vs = get_vectorstore()
        add_documents_with_bm25(chunks)
        logger.info(f"[Pipeline] Indexed {len(chunks)} chunks for document {document_id}")
    except Exception as e:
        logger.exception(f"[Pipeline] Vectorstore indexing failed: {e}")
        await _update_document_status(
            db, document_id,
            status="failed",
            parse_error=f"Vectorstore indexing failed: {e}",
            ocr_applied=ocr_used,
            pii_entities_found=pii_json,
            chunking_strategy=chunk_strategy,
        )
        return {
            "document_id": document_id,
            "chunks_ingested": 0,
            "pii_found": len(pii_entities),
            "ocr_used": ocr_used,
            "status": "failed",
            "parse_error": str(e),
        }

    # ── 6. Update document record ──────────────────────────────────────────────
    await _update_document_status(
        db, document_id,
        status="ready",
        ocr_applied=ocr_used,
        pii_entities_found=pii_json,
        chunking_strategy=chunk_strategy,
    )

    # ── 7. Done ─────────────────────────────────────────────────────────────────
    logger.info(
        f"[Pipeline] Complete — doc={document_id}, chunks={len(chunks)}, "
        f"pii={len(pii_entities)}, ocr={ocr_used}"
    )

    return {
        "document_id": document_id,
        "chunks_ingested": len(chunks),
        "pii_found": len(pii_entities),
        "ocr_used": ocr_used,
        "status": "ready",
        "parse_error": None,
    }


# ── Helper ────────────────────────────────────────────────────────────────────


async def _update_document_status(
    db: Optional[AsyncSession],
    document_id: str,
    status: str,
    parse_error: Optional[str] = None,
    ocr_applied: bool = False,
    pii_entities_found: str = "[]",
    chunking_strategy: str = "recursive",
) -> None:
    """Update a Document record's status fields."""
    if db is None:
        logger.warning("_update_document_status called with no db session")
        return

    try:
        # Import here to avoid circular
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            logger.warning(f"Document {document_id} not found for status update")
            return

        doc.status = status
        if parse_error is not None:
            doc.parse_error = parse_error
        doc.ocr_applied = ocr_applied
        doc.pii_entities_found = pii_entities_found
        doc.chunking_strategy = chunking_strategy

        await db.commit()
    except Exception as e:
        logger.exception(f"Failed to update document {document_id} status: {e}")
        await db.rollback()
