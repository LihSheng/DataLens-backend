"""
Frontend-aligned MVP API router.

Scope:
- Chat + conversations
- Documents + ACL + versions
- Settings
- Share + export
- FE-compatible feedback submit/stats
"""
from __future__ import annotations

import asyncio
import logging
logger = logging.getLogger(__name__)
import json
import mimetypes
import os
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from opentelemetry import trace as otel_trace

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.chains.rag_chain import RAGChain
from app.services.phoenix_annotations import run_live_ragas_eval, submit_feedback
from app.config import settings
from app.db.session import get_db
from app.dependencies import get_current_user
from app.export.markdown import conversation_to_markdown
from app.models.app_setting import AppSetting
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentAcl
from app.models.feedback import Feedback
from app.models.share_token import ShareToken
from app.models.user import User

router = APIRouter()


DEFAULT_RAG_SETTINGS: dict[str, Any] = {
    "modelName": "gpt-4o-mini",
    "topK": 5,
    "temperature": 0.7,
    "maxTokens": 2048,
    "showSourcesPanel": True,
    "enableStreaming": True,
    "hybridWeightDense": 0.5,
    "rerankerEnabled": False,
    "queryExpansionEnabled": False,
    "hydeEnabled": False,
    "chunkingStrategy": "semantic",
    "confidenceThreshold": 0.5,
    "memoryWindow": 5,
    "conversationRetentionDays": 30,
}


class ChatFilters(BaseModel):
    document_ids: list[str] | None = None
    doc_type: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    message: str
    conversation_id: str | None = Field(default=None, alias="conversationId")
    filters: ChatFilters | None = None


class RenameConversationRequest(BaseModel):
    title: str


class CreateConversationRequest(BaseModel):
    title: str | None = None


class DocumentAclPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    document_id: str | None = Field(default=None, alias="documentId")
    access_mode: str = Field(default="all", alias="accessMode")
    allowed_roles: list[str] = Field(default_factory=list, alias="allowedRoles")
    allowed_users: list[str] = Field(default_factory=list, alias="allowedUsers")


class FeedbackPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    message_id: str = Field(alias="messageId")
    conversation_id: str = Field(alias="conversationId")
    trace_id: str = Field(alias="traceId")
    rating: str
    comment: str | None = None


def _iso(dt: datetime | None) -> str:
    return dt.isoformat() if dt else datetime.utcnow().isoformat()


def _build_source(doc: Any, index: int) -> dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    source_name = metadata.get("filename") or metadata.get("document_name") or metadata.get("source") or "Unknown"
    document_id = str(metadata.get("document_id") or metadata.get("source") or f"doc_{index + 1}")
    out = {
        "documentId": document_id,
        "documentName": str(source_name),
        "chunkText": getattr(doc, "page_content", "") or "",
        "relevanceScore": float(metadata.get("score", 0.7)),
    }
    page_number = metadata.get("page") or metadata.get("page_number")
    if page_number is not None:
        out["pageNumber"] = int(page_number)
    rerank_score = metadata.get("rerank_score")
    if rerank_score is not None:
        out["rerankScore"] = float(rerank_score)
    return out


def _message_to_wire(msg: Message) -> dict[str, Any]:
    data = {
        "id": msg.id,
        "conversationId": msg.conversation_id,
        "role": msg.role,
        "content": msg.content,
        "createdAt": _iso(msg.created_at),
    }
    if msg.metadata_json:
        try:
            meta = json.loads(msg.metadata_json)
            if isinstance(meta, dict):
                data.update(meta)
        except Exception:
            pass
    return data


def _serialise_document(doc: Document, restricted: bool = False) -> dict[str, Any]:
    mime_type = mimetypes.types_map.get(doc.extension or "", "application/octet-stream")
    pii_entities: list[str] = []
    if doc.pii_entities_found:
        try:
            parsed = json.loads(doc.pii_entities_found)
            if isinstance(parsed, list):
                pii_entities = [str(x) for x in parsed]
        except Exception:
            pii_entities = []
    return {
        "id": doc.id,
        "name": doc.name,
        "size": doc.size,
        "mimeType": mime_type,
        "status": doc.status,
        "uploadedAt": _iso(doc.created_at),
        "extension": doc.extension,
        "chunkCount": int(getattr(doc, "chunk_count", 0) or 0),
        "parseError": doc.parse_error,
        "ocrApplied": bool(doc.ocr_applied),
        "piiEntitiesFound": pii_entities,
        "version": int(doc.version or 1),
        "restricted": restricted,
    }


async def _run_inline_ingestion(
    *,
    file_path: str,
    user_id: str,
    document_id: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    redact_pii: bool,
    db: AsyncSession,
) -> dict[str, Any]:
    from app.ingestion.pipeline import run_ingestion_pipeline

    return await run_ingestion_pipeline(
        file_path=file_path,
        user_id=user_id,
        document_id=document_id,
        options={
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "redact_pii": redact_pii,
            "use_presidio": False,
            "enable_semantic": False,
        },
        db=db,
    )


async def _get_or_create_settings(db: AsyncSession, user_id: str) -> dict[str, Any]:
    row = (
        await db.execute(select(AppSetting).where(AppSetting.user_id == user_id))
    ).scalar_one_or_none()
    if row is None:
        row = AppSetting(user_id=user_id, data_json=json.dumps(DEFAULT_RAG_SETTINGS))
        db.add(row)
        try:
            await db.commit()
            await db.refresh(row)
            return dict(DEFAULT_RAG_SETTINGS)
        except IntegrityError:
            # Concurrent first-load race: another request created the settings row.
            await db.rollback()
            row = (
                await db.execute(select(AppSetting).where(AppSetting.user_id == user_id))
            ).scalar_one_or_none()
            if row is None:
                raise
    try:
        parsed = json.loads(row.data_json)
    except Exception:
        parsed = {}
    merged = dict(DEFAULT_RAG_SETTINGS)
    if isinstance(parsed, dict):
        merged.update(parsed)
    return merged


async def _save_settings(db: AsyncSession, user_id: str, data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_RAG_SETTINGS)
    merged.update(data)
    row = (
        await db.execute(select(AppSetting).where(AppSetting.user_id == user_id))
    ).scalar_one_or_none()
    if row is None:
        row = AppSetting(user_id=user_id, data_json=json.dumps(merged))
        db.add(row)
    else:
        row.data_json = json.dumps(merged)
    try:
        await db.commit()
    except IntegrityError:
        # Concurrent upsert race during first save.
        await db.rollback()
        row = (
            await db.execute(select(AppSetting).where(AppSetting.user_id == user_id))
        ).scalar_one_or_none()
        if row is None:
            raise
        row.data_json = json.dumps(merged)
        await db.commit()
    return merged


async def _ensure_conversation_owned(
    db: AsyncSession, conversation_id: str, user_id: str
) -> Conversation:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _snippet(text: str, q: str, max_len: int = 140) -> str:
    if not text:
        return ""
    text_l = text.lower()
    q_l = q.lower()
    idx = text_l.find(q_l)
    if idx < 0:
        return text[:max_len]
    start = max(0, idx - max_len // 3)
    end = min(len(text), start + max_len)
    s = text[start:end]
    if start > 0:
        s = "…" + s
    if end < len(text):
        s += "…"
    return s


@router.get("/settings")
async def get_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await _get_or_create_settings(db, current_user.id)


@router.post("/settings")
async def update_settings(
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = await _get_or_create_settings(db, current_user.id)
    existing.update(payload or {})
    return await _save_settings(db, current_user.id, existing)


@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    rows = result.scalars().all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "createdAt": _iso(c.created_at),
            "updatedAt": _iso(c.updated_at),
        }
        for c in rows
    ]


@router.post("/conversations")
async def create_conversation(
    payload: CreateConversationRequest | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    title = (payload.title if payload else None) or "New conversation"
    conv = Conversation(user_id=current_user.id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return {
        "id": conv.id,
        "title": conv.title,
        "createdAt": _iso(conv.created_at),
        "updatedAt": _iso(conv.updated_at),
    }


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    payload: RenameConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await _ensure_conversation_owned(db, conversation_id, current_user.id)
    conv.title = payload.title.strip() or conv.title
    conv.updated_at = datetime.utcnow()
    await db.commit()
    return {
        "id": conv.id,
        "title": conv.title,
        "createdAt": _iso(conv.created_at),
        "updatedAt": _iso(conv.updated_at),
    }


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await _ensure_conversation_owned(db, conversation_id, current_user.id)
    # Explicitly delete children before the parent to avoid DB FK constraint issues
    await db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await db.execute(delete(ShareToken).where(ShareToken.conversation_id == conversation_id))
    await db.delete(conv)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _ensure_conversation_owned(db, conversation_id, current_user.id)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    rows = result.scalars().all()
    return [_message_to_wire(m) for m in rows]


@router.get("/conversations/search")
async def search_conversations(
    q: str = Query("", min_length=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = q.strip()
    if not q:
        return []

    like = f"%{q}%"
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .where(
            or_(
                Conversation.title.ilike(like),
                Conversation.id.in_(
                    select(Message.conversation_id).where(Message.content.ilike(like))
                ),
            )
        )
        .order_by(Conversation.updated_at.desc())
        .limit(20)
    )
    result = await db.execute(stmt)
    conversations = result.scalars().all()

    out: list[dict[str, Any]] = []
    for conv in conversations:
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv.id, Message.content.ilike(like))
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        matched = msg_result.scalar_one_or_none()
        out.append(
            {
                "id": conv.id,
                "title": conv.title,
                "createdAt": _iso(conv.created_at),
                "updatedAt": _iso(conv.updated_at),
                "snippet": _snippet(matched.content if matched else conv.title, q),
            }
        )
    return out


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("rag_request"):
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        conversation_id = payload.conversation_id
        if not conversation_id or conversation_id == "conv_new":
            conv = Conversation(user_id=current_user.id, title="New conversation")
            db.add(conv)
            await db.commit()
            await db.refresh(conv)
            conversation_id = conv.id
        else:
            conv = await _ensure_conversation_owned(db, conversation_id, current_user.id)

        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=payload.message,
            metadata_json=None,
        )
        db.add(user_message)
        await db.commit()

        answer = ""
        source_docs: list[Any] = []
        followups: list[str] = []
        cache_hit = False
        routed_model = None
        input_tokens = 0
        output_tokens = 0
        error_code: str | None = None

        try:
            chain_filters = payload.filters.model_dump(exclude_none=True) if payload.filters else {}
            chain = RAGChain(
                settings={
                    "query_expansion": False,
                    "hyde": False,
                    "reranker": False,
                    "context_max_tokens": 1800,
                },
                filters=chain_filters,
                conversation_id=conversation_id,
            )
            result = chain.invoke({"question": payload.message})
            answer = str(result.get("answer", "")).strip()
            source_docs = result.get("source_documents", []) or []
            followups = result.get("followup_questions", []) or []
            cache_hit = bool(result.get("cache_hit", False))
            routed_model = result.get("model")
            input_tokens = int(result.get("input_tokens", 0))
            output_tokens = int(result.get("output_tokens", 0))
        except Exception:
            logger.exception("RAG inference failed in /api/chat")
            error_code = "RAG_INFERENCE_FAILED"
            answer = (
                "I couldn't answer right now due to a server error during retrieval or inference. "
                "Please try again."
            )

        if not answer:
            answer = "I couldn't find enough context to answer confidently."

        sources = [_build_source(doc, i) for i, doc in enumerate(source_docs)]
        token_usage = {
            "used": max(0, input_tokens),
            "available": int((await _get_or_create_settings(db, current_user.id)).get("maxTokens", 2048)),
            "chunksIncluded": len(sources),
            "chunksAvailable": len(sources),
        }

        assistant_meta = {
            "sources": sources,
            "suggestedFollowups": followups[:3],
            "confidence": "high" if sources else "low",
            "cacheHit": cache_hit,
            "routedToModel": routed_model,
            "latencyMs": 0,
            "tokenUsage": token_usage,
        }
        if error_code:
            assistant_meta["errorCode"] = error_code

        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            metadata_json=json.dumps(assistant_meta),
        )
        db.add(assistant_message)
        conv.updated_at = datetime.utcnow()
        await db.commit()

        # Fire-and-forget RAGAS eval + Phoenix annotation after db commit
        asyncio.create_task(
            run_live_ragas_eval(
                question=payload.message,
                answer=answer,
                source_docs=source_docs,
                trace_id=conversation_id,
            )
        )

        async def event_stream():
            yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"
            if sources:
                yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
            meta_payload: dict[str, Any] = {
                "conversationId": conversation_id,
                "tokenUsage": token_usage,
                "cacheHit": cache_hit,
            }
            if error_code:
                meta_payload["errorCode"] = error_code
            if followups:
                meta_payload["suggestedFollowups"] = followups[:3]
            if routed_model:
                meta_payload["routedToModel"] = routed_model
            yield f"data: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"
            yield "data: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


@router.get("/documents")
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()

    restricted_map: dict[str, bool] = {}
    if docs:
        acl_rows = await db.execute(
            select(DocumentAcl).where(DocumentAcl.document_id.in_([d.id for d in docs]))
        )
        by_doc: dict[str, list[DocumentAcl]] = {}
        for row in acl_rows.scalars().all():
            by_doc.setdefault(row.document_id, []).append(row)
        for d in docs:
            acl = by_doc.get(d.id, [])
            if not acl:
                restricted_map[d.id] = False
                continue
            role_allowed = any(
                a.principal_type == "role"
                and a.principal_id == current_user.role
                and a.can_read
                for a in acl
            )
            user_allowed = any(
                a.principal_type == "user"
                and a.principal_id == current_user.id
                and a.can_read
                for a in acl
            )
            restricted_map[d.id] = not (role_allowed or user_allowed)

    return [_serialise_document(d, restricted=restricted_map.get(d.id, False)) for d in docs]


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lower()
    out_path = upload_dir / f"{uuid.uuid4()}{ext}"

    contents = await file.read()
    with open(out_path, "wb") as f:
        f.write(contents)

    doc = Document(
        user_id=current_user.id,
        name=file.filename,
        file_path=str(out_path),
        size=len(contents),
        extension=ext,
        status="processing",
        chunking_strategy="recursive",
        version=1,
        is_active_version=True,
        pii_entities_found="[]",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    if settings.vectorstore_type == "memory":
        result = await _run_inline_ingestion(
            file_path=str(out_path),
            user_id=current_user.id,
            document_id=doc.id,
            chunk_strategy="recursive",
            chunk_size=1000,
            chunk_overlap=200,
            redact_pii=False,
            db=db,
        )
        await db.refresh(doc)
        logger.info(
            f"[API] Ingested document {doc.id} inline, status={result.get('status')}"
        )
    else:
        # Dispatch Celery ingest task
        try:
            from app.workers.ingestion_worker import process_document

            process_document.delay(
                file_path=str(out_path),
                user_id=current_user.id,
                document_id=doc.id,
                options={
                    "chunk_strategy": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "redact_pii": False,
                    "use_presidio": False,
                    "enable_semantic": False,
                },
            )
            logger.info(f"[API] Queued document {doc.id} for processing")
        except Exception as e:
            logger.exception(f"[API] Failed to dispatch Celery task: {e}")
            result = await _run_inline_ingestion(
                file_path=str(out_path),
                user_id=current_user.id,
                document_id=doc.id,
                chunk_strategy="recursive",
                chunk_size=1000,
                chunk_overlap=200,
                redact_pii=False,
                db=db,
            )
            await db.refresh(doc)
            logger.info(
                f"[API] Fallback inline ingestion completed for {doc.id}, "
                f"status={result.get('status')}"
            )

    return _serialise_document(doc, restricted=False)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    docs_to_delete: list[Document]
    if doc.parent_document_id is None:
        family_result = await db.execute(
            select(Document).where(
                Document.user_id == current_user.id,
                or_(Document.id == doc.id, Document.parent_document_id == doc.id),
            )
        )
        docs_to_delete = family_result.scalars().all()
    else:
        docs_to_delete = [doc]

    ids = [d.id for d in docs_to_delete]
    child_ids = [d.id for d in docs_to_delete if d.parent_document_id is not None]
    root_ids = [d.id for d in docs_to_delete if d.parent_document_id is None]
    file_paths = {d.file_path for d in docs_to_delete if d.file_path}

    if ids:
        await db.execute(delete(DocumentAcl).where(DocumentAcl.document_id.in_(ids)))
    if child_ids:
        await db.execute(delete(Document).where(Document.id.in_(child_ids)))
    if root_ids:
        await db.execute(delete(Document).where(Document.id.in_(root_ids)))
    await db.commit()
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.is_active_version = False
    new_doc = Document(
        user_id=doc.user_id,
        name=doc.name,
        file_path=doc.file_path,
        size=doc.size,
        extension=doc.extension,
        status="processing",
        parse_error=None,
        ocr_applied=doc.ocr_applied,
        pii_entities_found=doc.pii_entities_found or "[]",
        chunking_strategy=doc.chunking_strategy or "recursive",
        version=(doc.version or 1) + 1,
        parent_document_id=doc.parent_document_id or doc.id,
        is_active_version=True,
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    if settings.vectorstore_type == "memory":
        await _run_inline_ingestion(
            file_path=new_doc.file_path,
            user_id=new_doc.user_id,
            document_id=new_doc.id,
            chunk_strategy=new_doc.chunking_strategy or "recursive",
            chunk_size=1000,
            chunk_overlap=200,
            redact_pii=False,
            db=db,
        )
        await db.refresh(new_doc)
    else:
        try:
            from app.workers.ingestion_worker import process_document

            process_document.delay(
                file_path=new_doc.file_path,
                user_id=new_doc.user_id,
                document_id=new_doc.id,
                options={
                    "chunk_strategy": new_doc.chunking_strategy or "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "redact_pii": False,
                    "use_presidio": False,
                    "enable_semantic": False,
                },
            )
        except Exception as e:
            logger.exception(f"[API] Failed to dispatch Celery task: {e}")
            await _run_inline_ingestion(
                file_path=new_doc.file_path,
                user_id=new_doc.user_id,
                document_id=new_doc.id,
                chunk_strategy=new_doc.chunking_strategy or "recursive",
                chunk_size=1000,
                chunk_overlap=200,
                redact_pii=False,
                db=db,
            )
            await db.refresh(new_doc)
    return {"message": "Reindex queued", "documentId": new_doc.id}


@router.get("/documents/{document_id}/versions")
async def get_document_versions(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    root_id = doc.parent_document_id or doc.id
    rows = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            or_(Document.id == root_id, Document.parent_document_id == root_id),
        ).order_by(Document.version.asc())
    )
    versions = rows.scalars().all()
    return [
        {
            "id": v.id,
            "version": int(v.version or 1),
            "uploadedAt": _iso(v.created_at),
            "status": v.status,
            "isActive": bool(v.is_active_version),
        }
        for v in versions
    ]


@router.get("/documents/{document_id}/acl")
async def get_document_acl(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = await db.execute(select(DocumentAcl).where(DocumentAcl.document_id == document_id))
    acl = rows.scalars().all()
    if not acl:
        return {
            "documentId": document_id,
            "accessMode": "all",
            "allowedRoles": [],
            "allowedUsers": [],
        }

    roles = [r.principal_id for r in acl if r.principal_type == "role" and r.can_read]
    users = [r.principal_id for r in acl if r.principal_type == "user" and r.can_read]
    access_mode = "users" if users else "roles"
    return {
        "documentId": document_id,
        "accessMode": access_mode,
        "allowedRoles": roles,
        "allowedUsers": users,
    }


@router.put("/documents/{document_id}/acl")
async def update_document_acl(
    document_id: str,
    payload: DocumentAclPayload,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if payload.access_mode not in {"all", "roles", "users"}:
        raise HTTPException(status_code=400, detail="Invalid accessMode")

    await db.execute(delete(DocumentAcl).where(DocumentAcl.document_id == document_id))

    if payload.access_mode == "roles":
        for role in payload.allowed_roles:
            db.add(
                DocumentAcl(
                    document_id=document_id,
                    principal_type="role",
                    principal_id=role,
                    can_read=True,
                )
            )
    elif payload.access_mode == "users":
        for user_id in payload.allowed_users:
            db.add(
                DocumentAcl(
                    document_id=document_id,
                    principal_type="user",
                    principal_id=user_id,
                    can_read=True,
                )
            )
    await db.commit()

    return {
        "documentId": document_id,
        "accessMode": payload.access_mode,
        "allowedRoles": payload.allowed_roles,
        "allowedUsers": payload.allowed_users,
    }


@router.post("/conversations/{conversation_id}/share")
async def create_share_link(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await _ensure_conversation_owned(db, conversation_id, current_user.id)

    existing_result = await db.execute(
        select(ShareToken).where(
            ShareToken.conversation_id == conversation_id,
            ShareToken.is_active == True,  # noqa: E712
        )
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        return {"token": existing.token, "url": f"/share/{existing.token}"}

    token = secrets.token_urlsafe(24)
    share = ShareToken(
        conversation_id=conversation_id,
        token=token,
        created_by=current_user.id,
        expires_at=datetime.utcnow() + timedelta(days=30),
        is_active=True,
    )
    db.add(share)
    await db.commit()
    return {"token": token, "url": f"/share/{token}"}


@router.get("/share/{token}")
async def get_shared_conversation(token: str, db: AsyncSession = Depends(get_db)):
    token_result = await db.execute(
        select(ShareToken).where(
            ShareToken.token == token,
            ShareToken.is_active == True,  # noqa: E712
        )
    )
    share = token_result.scalar_one_or_none()
    if share is None:
        raise HTTPException(status_code=404, detail="Shared conversation not found")
    if share.expires_at and share.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Shared conversation expired")

    convo_result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == share.conversation_id)
    )
    conv = convo_result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = sorted(conv.messages, key=lambda m: m.created_at)
    return {
        "id": conv.id,
        "title": conv.title,
        "messages": [_message_to_wire(m) for m in messages],
        "createdAt": _iso(conv.created_at),
        "updatedAt": _iso(conv.updated_at),
    }


@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: str = Query("md"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await _ensure_conversation_owned(db, conversation_id, current_user.id)
    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = msg_result.scalars().all()
    markdown = conversation_to_markdown(conv, messages)
    media = "text/markdown; charset=utf-8"
    filename = f"conversation.{ 'pdf' if format == 'pdf' else 'md' }"
    return Response(
        content=markdown,
        media_type=media,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/feedback")
async def submit_feedback(
    payload: FeedbackPayload,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("feedback_submit"):
        vote = "positive" if payload.rating == "positive" else "negative"
        try:
            message_uuid = uuid.UUID(payload.message_id)
            user_uuid = uuid.UUID(current_user.id)
        except ValueError:
            raise HTTPException(status_code=400, detail="messageId and userId must be UUIDs")

        feedback = Feedback(
            message_id=message_uuid,
            user_id=user_uuid,
            vote=vote,
            rating=5 if vote == "positive" else 1,
            comment=payload.comment,
            metadata_json=json.dumps({"traceId": payload.trace_id, "conversationId": payload.conversation_id}),
        )
        db.add(feedback)
        await db.commit()
        await db.refresh(feedback)
        # Fire-and-forget Phoenix human-feedback annotation
        asyncio.create_task(
            submit_feedback(
                trace_id=payload.trace_id or payload.conversation_id or "",
                span_id=None,
                label=vote,
            )
        )

    return {
        "id": str(feedback.id),
        "messageId": str(feedback.message_id),
        "userId": str(feedback.user_id),
        "rating": vote,
        "comment": feedback.comment,
        "createdAt": _iso(feedback.created_at),
    }


@router.get("/feedback/stats")
async def feedback_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # MVP global stats
    rows = (await db.execute(select(Feedback))).scalars().all()
    total = len(rows)
    positive = sum(1 for r in rows if r.vote == "positive")
    negative = sum(1 for r in rows if r.vote == "negative")
    positive_ratio = (positive / total) if total else 0.0
    negative_ratio = (negative / total) if total else 0.0
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positiveRatio": round(positive_ratio, 4),
        "negativeRatio": round(negative_ratio, 4),
        "trend": "stable",
    }
