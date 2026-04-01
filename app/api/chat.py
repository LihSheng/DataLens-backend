"""
Chat API — SSE streaming + RAG query endpoints.
Stage 1: retrieval pipeline via RAGChain (hybrid + rerank + filters).

Matches frontend expectations:
- POST /api/chat — streaming SSE, accepts { message, conversationId?, filters? }
- POST /api/query — plain JSON, accepts { question, k, settings?, filters? }
"""
from typing import List, Optional, AsyncIterator
import json
import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chains.rag_chain import RAGChain
from app.config import settings as app_settings

router = APIRouter()


# ─────────────────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────────────────

class ChatFilters(BaseModel):
    document_ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Frontend-facing chat request — uses 'message' field."""
    message: str
    conversation_id: Optional[str] = None
    filters: Optional[ChatFilters] = None


class ChatResponse(BaseModel):
    """Frontend-facing plain JSON response (non-streaming)."""
    answer: str
    sources: List[str]
    conversation_id: Optional[str] = None


class QuerySettings(BaseModel):
    query_expansion: Optional[bool] = None
    hyde: Optional[bool] = None
    reranker: Optional[bool] = None


class QueryFilters(BaseModel):
    document_ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class QueryRequest(BaseModel):
    """Internal /query endpoint — uses 'question' field."""
    question: str
    k: int = 8
    rerank_top_n: int = 4
    settings: Optional[QuerySettings] = None
    filters: Optional[QueryFilters] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _build_rag_chain(req_settings: dict, req_filters: dict = None) -> RAGChain:
    """Build RAGChain with merged settings (app defaults + request overrides)."""
    chain_settings = {
        "query_expansion": app_settings.query_expansion_enabled,
        "hyde": app_settings.hyde_enabled,
        "reranker": app_settings.reranker_enabled,
        "confidence_threshold": app_settings.confidence_threshold,
    }
    chain_settings.update({k: v for k, v in req_settings.items() if v is not None})

    filters = req_filters or {}
    filters = {k: v for k, v in filters.items() if v is not None}

    return RAGChain(settings=chain_settings, filters=filters or None)


def _format_sources(docs) -> List[str]:
    return [
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content[:200]}"
        for doc in docs
    ]


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest):
    """
    SSE streaming chat endpoint — matches frontend's sendMessage().
    FE calls POST /api/chat with { message, conversationId?, filters? }
    """
    chain = _build_rag_chain({}, req.filters and req.filters.model_dump(exclude_none=True))

    async def event_stream() -> AsyncIterator[str]:
        result = chain.invoke({"question": req.message})
        answer = result["answer"]
        sources = _format_sources(result.get("source_documents", []))

        # Send answer
        yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
        # Send sources
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
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
    Full retrieval pipeline: HybridRetriever + optional QueryExpander + HyDE + Reranker.
    """
    chain_settings = req.settings.model_dump(exclude_none=True) if req.settings else {}
    chain_filters = req.filters.model_dump(exclude_none=True) if req.filters else None

    chain = _build_rag_chain(chain_settings, chain_filters)
    result = chain.invoke({"question": req.question})

    return QueryResponse(
        answer=result["answer"],
        sources=_format_sources(result.get("source_documents", [])),
    )
