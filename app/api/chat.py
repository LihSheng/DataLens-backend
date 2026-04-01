"""
Chat API — SSE streaming + RAG query endpoints.
Stage 1: retrieval pipeline via RAGChain (hybrid + rerank + filters).
Stage 3: SafetyLayer (guardrails, prompt injection, grounding, citations).
Stage 4: ConversationMemory (Redis), follow-up questions, trace_id propagation.

Matches frontend expectations:
- POST /api/chat — streaming SSE, accepts { message, conversationId?, filters? }
- POST /api/query — plain JSON, accepts { question, conversationId?, k, settings?, filters? }
"""
import uuid
from typing import List, Optional, AsyncIterator

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chains.rag_chain import RAGChain
from app.config import settings as app_settings
from app.memory.conversation_memory import (
    add_user_message,
    add_assistant_message,
    conversation_exists,
    get_conversation_memory,
)

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


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str
    trace_id: str
    fallback_triggered: bool = False
    grounded: Optional[bool] = None
    citations_valid: Optional[bool] = None
    followup_questions: List[str] = []


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _ensure_conversation_id(conversation_id: Optional[str]) -> str:
    """Return existing ID or generate a new one."""
    if conversation_id and conversation_exists(conversation_id):
        return conversation_id
    return str(uuid.uuid4())


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
async def chat(req: ChatRequest):
    """
    SSE streaming chat endpoint — matches frontend's sendMessage().
    Also stores messages in Redis ConversationMemory and returns
    trace_id + follow-up questions (Stage 4).
    """
    # Ensure conversation exists
    conversation_id = _ensure_conversation_id(req.conversation_id)

    chain_settings = req.filters and req.filters.model_dump(exclude_none=True) or {}
    chain = _build_rag_chain({}, chain_settings, conversation_id=conversation_id)

    async def event_stream() -> AsyncIterator[str]:
        result = chain.invoke({"question": req.message})
        answer = result["answer"]
        sources = _format_sources(result.get("source_documents", []))
        trace_id = result.get("trace_id", "")
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

        # Stream answer
        yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
        # Stream sources
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        # Stream trace_id
        yield f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id})}\n\n"
        # Stream follow-up questions
        if followups:
            yield f"data: {json.dumps({'type': 'followup_questions', 'questions': followups})}\n\n"
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
def query(req: QueryRequest):
    """
    Plain JSON query endpoint for internal/testing use.
    Full retrieval pipeline + Stage 3 safety + Stage 4 memory.
    """
    conversation_id = _ensure_conversation_id(req.conversation_id)

    chain_settings = req.settings.model_dump(exclude_none=True) if req.settings else {}
    chain_filters = req.filters.model_dump(exclude_none=True) if req.filters else None

    chain = _build_rag_chain(chain_settings, chain_filters, conversation_id=conversation_id)
    result = chain.invoke({"question": req.question})

    safety_response = result.get("safety_response")

    # Persist messages (fire-and-forget in query mode)
    trace_id = result.get("trace_id", "")
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

    return QueryResponse(
        answer=result["answer"],
        sources=_format_sources(result.get("source_documents", [])),
        conversation_id=conversation_id,
        trace_id=trace_id,
        fallback_triggered=result.get("fallback_triggered", False),
        grounded=safety_response.grounded if safety_response else None,
        citations_valid=safety_response.citations_valid if safety_response else None,
        followup_questions=result.get("followup_questions", []),
    )
