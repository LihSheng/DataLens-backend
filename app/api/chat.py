"""
Chat API — SSE streaming + RAG query endpoints.
Stage 1: retrieval pipeline via RAGChain (hybrid + rerank + filters).
Stage 3: SafetyLayer (guardrails, prompt injection, grounding, citations).
Stage 3: RAGAS live eval + human feedback (Phoenix span annotations).
Stage 4: ConversationMemory (Redis), follow-up questions, trace_id propagation.
Stage 5: ChatResponse contract — trace_metadata field + SSE event.

Matches frontend expectations:
- POST /api/chat — streaming SSE, accepts { message, conversationId?, filters? }
- POST /api/query — plain JSON, accepts { question, conversationId?, k, settings?, filters? }
- POST /api/feedback — human feedback (positive/negative), forwarded to Phoenix.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional, AsyncIterator

import json

from opentelemetry import trace as otel_trace

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.chains.rag_chain import RAGChain
from app.config import settings as app_settings
from app.db.session import get_db
from app.memory.conversation_memory import (
    add_user_message,
    add_assistant_message,
    get_conversation_memory,
)
from app.services.cost_tracker import record_query_cost
from app.services.phoenix_annotations import run_live_ragas_eval, submit_feedback
from app.api._errors import build_error

router = APIRouter()


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class ChatFilters(BaseModel):
    document_ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Frontend-facing chat request."""
    message: str
    conversation_id: Optional[str] = None
    filters: Optional[ChatFilters] = None


class QuerySettings(BaseModel):
    # Retrieval
    query_expansion: Optional[bool] = None
    hyde: Optional[bool] = None
    reranker: Optional[bool] = None
    # Safety + quality (Stage 3)
    guardrails_enabled: Optional[bool] = None
    prompt_injection_check: Optional[bool] = None
    grounding_check: Optional[bool] = None
    citation_validation: Optional[bool] = None
    grounding_threshold: Optional[float] = None
    injection_confidence_threshold: Optional[float] = None
    required_citation_threshold: Optional[float] = None
    max_retries: Optional[int] = None
    # Memory (Stage 4)
    enable_memory: Optional[bool] = True
    enable_followup: Optional[bool] = True
    conversation_history_limit: Optional[int] = 10
    # Stage 5
    semantic_cache_enabled: Optional[bool] = True
    semantic_cache_threshold: Optional[float] = None
    context_max_tokens: Optional[int] = None
    routing_mode: Optional[str] = None
    fast_model: Optional[str] = None
    quality_model: Optional[str] = None


class QueryFilters(BaseModel):
    document_ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class QueryRequest(BaseModel):
    """Internal /query endpoint."""
    question: str
    conversation_id: Optional[str] = None
    k: int = 8
    rerank_top_n: int = 4
    settings: Optional[QuerySettings] = None
    filters: Optional[QueryFilters] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str
    trace_id: str
    followup_questions: List[str] = []
    trace_metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str
    trace_id: str
    fallback_triggered: bool = False
    grounded: Optional[bool] = None
    citations_valid: Optional[bool] = None
    followup_questions: List[str] = []
    cache_hit: bool = False
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    trace_metadata: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _ensure_conversation_id(conversation_id: Optional[str]) -> str:
    """Return existing ID or generate a new one."""
    if conversation_id and get_conversation_memory().conversation_exists(conversation_id):
        return conversation_id
    return str(uuid.uuid4())


def _get_otel_trace_id(fallback_trace_id: str = "") -> str:
    """Extract real trace_id from OTel context, falling back to provided value."""
    ctx = otel_trace.get_current_span().get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return fallback_trace_id


def _build_rag_chain(
    req_settings: dict,
    req_filters: dict = None,
    conversation_id: Optional[str] = None,
) -> RAGChain:
    """Build RAGChain with merged settings + Stage 4 memory params."""
    chain_settings = {
        "query_expansion": app_settings.query_expansion_enabled,
        "hyde": app_settings.hyde_enabled,
        "reranker": app_settings.reranker_enabled,
        "confidence_threshold": app_settings.confidence_threshold,
        # Stage 4 defaults
        "enable_memory": True,
        "enable_followup": True,
        "conversation_history_limit": 10,
        # Stage 5 defaults
        "semantic_cache_enabled": app_settings.semantic_cache_enabled,
        "semantic_cache_threshold": app_settings.semantic_cache_threshold,
        "context_max_tokens": app_settings.context_max_tokens,
        "routing_mode": app_settings.routing_mode,
        "fast_model": app_settings.fast_model,
        "quality_model": app_settings.quality_model,
    }
    chain_settings.update({k: v for k, v in req_settings.items() if v is not None})

    filters = req_filters or {}
    filters = {k: v for k, v in filters.items() if v is not None}

    return RAGChain(
        settings=chain_settings,
        filters=filters or None,
        conversation_id=conversation_id,
        conversation_history_limit=chain_settings.get("conversation_history_limit", 10),
        enable_memory=chain_settings.get("enable_memory", True),
        enable_followup=chain_settings.get("enable_followup", True),
    )


def _format_sources(docs) -> List[str]:
    return [
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content[:200]}"
        for doc in docs
    ]


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    SSE streaming chat endpoint — matches frontend's sendMessage().
    Also stores messages in Redis ConversationMemory and returns
    trace_id + follow-up questions (Stage 4).
    """
    # Ensure conversation exists
    conversation_id = _ensure_conversation_id(req.conversation_id)

    chain_filters = req.filters.model_dump(exclude_none=True) if req.filters else None
    chain = _build_rag_chain({}, chain_filters, conversation_id=conversation_id)

    # Root span for the RAG request
    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("rag_request") as root_span:
        # TODO: wire from auth context (Stage 4+)
        root_span.set_attribute("user.id", "anonymous")
        root_span.set_attribute("conversation.id", conversation_id)
        root_span.set_attribute("query.length", len(req.message))

        async def event_stream() -> AsyncIterator[str]:
            result = chain.invoke({"question": req.message})
            answer = result["answer"]
            sources = _format_sources(result.get("source_documents", []))
            # Extract real trace_id from OTel context
            trace_id = _get_otel_trace_id(result.get("trace_id", ""))
            followups = result.get("followup_questions", [])

            # Persist user message
            add_user_message(conversation_id, req.message, trace_id=trace_id)

            # Persist assistant response with safety metadata
            metadata = {}
            if result.get("safety_response"):
                sr = result["safety_response"]
                metadata = {
                    "fallback_triggered": sr.fallback_triggered,
                    "grounded": sr.grounded,
                    "citations_valid": sr.citations_valid,
                }
            add_assistant_message(conversation_id, answer, metadata=metadata, trace_id=trace_id)

            # Persist QueryCost (best-effort)
            try:
                await record_query_cost(
                    db=db,
                    user_id="anonymous",
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                    model=result.get("model", ""),
                    input_tokens=int(result.get("input_tokens", 0)),
                    output_tokens=int(result.get("output_tokens", 0)),
                    cost_usd=float(result.get("cost_usd", 0.0)),
                )
            except Exception:
                pass

            # Stream answer
            yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
            # Stream sources
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            # Stream trace_id
            yield f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id})}\n\n"
            # Stream follow-up questions
            if followups:
                yield f"data: {json.dumps({'type': 'followup_questions', 'questions': followups})}\n\n"
            # Stream performance/cost metadata
            yield f"data: {json.dumps({'type': 'performance', 'cache_hit': bool(result.get('cache_hit', False)), 'model': result.get('model'), 'input_tokens': int(result.get('input_tokens', 0)), 'output_tokens': int(result.get('output_tokens', 0)), 'cost_usd': float(result.get('cost_usd', 0.0))})}\n\n"
            # Stream trace metadata (Stage 5)
            meta = {
                'traceId': trace_id,
                'tokens': int(result.get('input_tokens', 0)) + int(result.get('output_tokens', 0)),
                'retrievedChunks': len(result.get('source_documents', [])),
                'usedChunks': len(result.get('source_documents', [])),
            }
            yield f"data: {json.dumps({'type': 'trace_metadata', 'metadata': meta})}\n\n"
            # Fire-and-forget RAGAS live evaluation (Stage 3)
            asyncio.create_task(
                run_live_ragas_eval(
                    question=req.message,
                    answer=answer,
                    source_docs=result.get("source_documents", []),
                    trace_id=trace_id,
                )
            )
            # Done
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Plain JSON query endpoint for internal/testing use.
    Full retrieval pipeline + Stage 3 safety + Stage 4 memory.
    """
    conversation_id = _ensure_conversation_id(req.conversation_id)

    # Root span for the RAG request
    tracer = otel_trace.get_tracer(__name__)
    with tracer.start_as_current_span("rag_request") as root_span:
        # TODO: wire from auth context (Stage 4+)
        root_span.set_attribute("user.id", "anonymous")
        root_span.set_attribute("conversation.id", conversation_id)
        root_span.set_attribute("query.length", len(req.question))

        chain_settings = req.settings.model_dump(exclude_none=True) if req.settings else {}
        chain_filters = req.filters.model_dump(exclude_none=True) if req.filters else None

        chain = _build_rag_chain(chain_settings, chain_filters, conversation_id=conversation_id)
        result = chain.invoke({"question": req.question})

        safety_response = result.get("safety_response")

        # Extract real trace_id from OTel context
        trace_id = _get_otel_trace_id(result.get("trace_id", ""))

        # Persist messages (fire-and-forget in query mode)
        try:
            add_user_message(conversation_id, req.question, trace_id=trace_id)
            metadata = {}
            if safety_response:
                metadata = {
                    "fallback_triggered": safety_response.fallback_triggered,
                    "grounded": safety_response.grounded,
                    "citations_valid": safety_response.citations_valid,
                }
            add_assistant_message(conversation_id, result["answer"], metadata=metadata, trace_id=trace_id)
        except Exception:
            pass  # Non-fatal in query mode

        # Persist QueryCost (best-effort)
        try:
            await record_query_cost(
                db=db,
                user_id="anonymous",
                conversation_id=conversation_id,
                trace_id=trace_id,
                model=result.get("model", ""),
                input_tokens=int(result.get("input_tokens", 0)),
                output_tokens=int(result.get("output_tokens", 0)),
                cost_usd=float(result.get("cost_usd", 0.0)),
            )
        except Exception:
            pass

        return QueryResponse(
            answer=result["answer"],
            sources=_format_sources(result.get("source_documents", [])),
            conversation_id=conversation_id,
            trace_id=trace_id,
            fallback_triggered=result.get("fallback_triggered", False),
            grounded=safety_response.grounded if safety_response else None,
            citations_valid=safety_response.citations_valid if safety_response else None,
            followup_questions=result.get("followup_questions", []),
            cache_hit=bool(result.get("cache_hit", False)),
            model=result.get("model"),
            input_tokens=int(result.get("input_tokens", 0)),
            output_tokens=int(result.get("output_tokens", 0)),
            cost_usd=float(result.get("cost_usd", 0.0)),
            trace_metadata={
                "traceId": trace_id,
                "tokens": int(result.get("input_tokens", 0)) + int(result.get("output_tokens", 0)),
                "retrievedChunks": len(result.get("source_documents", [])),
                "usedChunks": len(result.get("source_documents", [])),
            },
        )

        # Fire-and-forget RAGAS live evaluation (Stage 3)
        asyncio.create_task(
            run_live_ragas_eval(
                question=req.question,
                answer=result["answer"],
                source_docs=result.get("source_documents", []),
                trace_id=trace_id,
            )
        )


# ─────────────────────────────────────────────────────────
# Feedback endpoint (Stage 3)
# ─────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    trace_id: str
    span_id: Optional[str] = None
    label: str  # "positive" or "negative"


class FeedbackResponse(BaseModel):
    success: bool
    trace_id: str
    label: str


@router.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(req: FeedbackRequest):
    """
    Record human feedback (thumbs up/down) for a chat response.

    Feedback is forwarded to Phoenix as a span annotation (Stage 3).
    This endpoint is public (not admin-gated) so end users can submit feedback.
    """
    if req.label not in ("positive", "negative"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=build_error(code="INVALID_LABEL", message="label must be 'positive' or 'negative'"),
        )

    success = await submit_feedback(
        trace_id=req.trace_id,
        span_id=req.span_id,
        label=req.label,
    )

    return FeedbackResponse(success=success, trace_id=req.trace_id, label=req.label)
