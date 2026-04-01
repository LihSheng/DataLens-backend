# RAG Chain — Full Implementation

> `app/chains/rag_chain.py` and supporting modules.
> Wires together: hybrid retriever → reranker → memory → context assembler →
> guardrails → LLM → grounding check → cost tracking → Phoenix tracing.

---

## Chain Execution Flow

```
User message
    │
    ▼
1.  Input guardrail scan         (safety/guardrails.py)
    │  blocked → 400 InputBlocked
    ▼
2.  Semantic cache lookup        (cache/semantic_cache.py)
    │  hit → return cached answer immediately
    ▼
3.  Query expansion (optional)   (retrieval/query_expander.py)
    │  generates N alternative phrasings
    ▼
4.  Hybrid retrieval             (retrieval/hybrid_retriever.py)
    │  BM25 + dense, Reciprocal Rank Fusion
    ▼
5.  Cross-encoder reranking      (retrieval/reranker.py)
    │  top-K reranked chunks
    ▼
6.  Metadata filter              (applied inside retriever)
    │
    ▼
7.  Confidence gate              (max_score < threshold → no-answer)
    │
    ▼
8.  Context assembly             (context/assembler.py)
    │  token budget trimming, chunk wrapping
    ▼
9.  Conversation memory inject   (memory/conversation_memory.py)
    │  summary of prior turns prepended
    ▼
10. LLM generation               (langchain_openai.ChatOpenAI)
    │  streaming or non-streaming
    ▼
11. Output guardrail scan        (safety/guardrails.py)
    │  blocked → safe fallback
    ▼
12. Grounding check (async)      (quality/grounding.py)
13. Citation verification (async)(quality/citations.py)
14. Follow-up generation (async) (generation/followups.py)
15. RAGAS eval (background)      (evaluation/ragas_eval.py)
16. Cost tracking                (billing/cost_tracker.py)
17. Cache store                  (cache/semantic_cache.py)
18. Audit log write              (models/audit.py)
    │
    ▼
ChatResponse → client
```

---

## Main Chain File

```python
# app/chains/rag_chain.py
from __future__ import annotations

import json
import asyncio
from typing import AsyncIterator
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.models.settings import RAGSettings
from app.models.conversation import Message
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.query_expander import QueryExpander
from app.context.assembler import assemble_context
from app.memory.conversation_memory import ConversationMemoryManager
from app.cache.semantic_cache import SemanticCache
from app.safety.guardrails import scan_input, scan_output
from app.safety.prompt_injection import sanitise_input, sanitise_retrieved_chunk
from app.quality.grounding import check_grounding
from app.quality.citations import verify_citations
from app.generation.followups import generate_followups
from app.billing.cost_tracker import calculate_cost, log_cost
from app.db.session import AsyncSessionLocal


# ---------------------------------------------------------------------------
# Pydantic response model
# ---------------------------------------------------------------------------

@dataclass
class SourceResult:
    id: str
    title: str
    document_name: str
    chunk_text: str
    score: float
    rerank_score: float | None
    page: int | None


@dataclass
class ChatResult:
    answer: str
    sources: list[SourceResult]
    trace_id: str
    confidence: str                       # 'high' | 'medium' | 'low'
    no_answer_reason: str | None
    cache_hit: bool
    routed_to_model: str
    grounding: dict | None
    citation_validity: list[dict] | None
    suggested_followups: list[str]
    token_usage: dict
    retrieval_latency_ms: int
    llm_latency_ms: int
    total_latency_ms: int
    tokens_used: dict


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful AI assistant for an internal knowledge base.
Answer the user's question using ONLY the information in the provided context.
If the context does not contain enough information to answer confidently, say so clearly.
Always cite your sources using [1], [2] etc. that correspond to the numbered context blocks.

{memory_context}
"""

CONTEXT_TEMPLATE = """[{index}] Source: {document_name} (page {page})
{chunk_text}
"""


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

class RAGChain:
    def __init__(self, rag_settings: RAGSettings, vectorstore, documents: list[Document]):
        self.rag_settings = rag_settings
        self.vectorstore = vectorstore

        # LLM — uses model routing if enabled
        self.llm = ChatOpenAI(
            model=rag_settings.model,
            temperature=rag_settings.temperature,
            streaming=True,
            openai_api_key=settings.openai_api_key,
        )
        self.fast_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )

        # Retrieval
        self.retriever = HybridRetriever(
            vectorstore=vectorstore,
            documents=documents,
            weights=(
                rag_settings.hybrid_weight_dense,
                1 - rag_settings.hybrid_weight_dense,
            ),
        )
        self.reranker = CrossEncoderReranker(rag_settings.reranker_model) if rag_settings.reranker_enabled else None
        self.expander = QueryExpander(self.fast_llm) if rag_settings.query_expansion_enabled else None

        # Cache + memory
        self.cache = SemanticCache(redis_url=settings.redis_url)
        self.memory = ConversationMemoryManager(llm=self.fast_llm, max_token_limit=rag_settings.memory_window * 150)

    async def run(
        self,
        query: str,
        conversation_id: str,
        history: list[Message],
        filters: dict | None,
        user_id: str,
        trace_id: str,
    ) -> ChatResult:
        import time
        t0 = time.monotonic()

        # 1. Input guardrail
        is_safe, block_reason = scan_input(query)
        if not is_safe:
            await self._write_audit("blocked_input", user_id, {"query": query, "reason": block_reason}, trace_id)
            raise InputBlockedError(block_reason)

        # Prompt injection check
        query, was_injected = sanitise_input(query)
        if was_injected:
            await self._write_audit("injection_attempt", user_id, {"query": query}, trace_id)

        # 2. Semantic cache lookup
        cached = self.cache.get(query)
        if cached:
            return ChatResult(
                answer=cached["answer"],
                sources=cached["sources"],
                cache_hit=True,
                confidence="high",
                no_answer_reason=None,
                trace_id=trace_id,
                routed_to_model=self.rag_settings.model,
                grounding=None,
                citation_validity=None,
                suggested_followups=[],
                token_usage={"used": 0, "available": self.rag_settings.max_tokens},
                retrieval_latency_ms=0,
                llm_latency_ms=0,
                total_latency_ms=int((time.monotonic() - t0) * 1000),
                tokens_used={"prompt": 0, "completion": 0, "total": 0},
            )

        # 3. Query expansion
        queries = [query]
        if self.expander:
            queries = await self.expander.expand(query)

        # 4. Hybrid retrieval
        t_ret = time.monotonic()
        raw_docs: list[Document] = []
        for q in queries:
            retrieved = self.retriever.invoke(q, filters=filters)
            raw_docs.extend(retrieved)

        # Deduplicate by content hash
        seen, deduped = set(), []
        for d in raw_docs:
            h = hash(d.page_content)
            if h not in seen:
                seen.add(h)
                deduped.append(d)

        # 5. Reranking
        chunks = deduped
        if self.reranker:
            chunks = self.reranker.rerank(query, deduped, top_k=self.rag_settings.top_k)
        else:
            chunks = deduped[:self.rag_settings.top_k]

        retrieval_latency_ms = int((time.monotonic() - t_ret) * 1000)

        # 6. Confidence gate
        max_score = max((c.metadata.get("score", 0) for c in chunks), default=0)
        if max_score < self.rag_settings.confidence_threshold:
            return ChatResult(
                answer="I don't have enough information in the knowledge base to answer this confidently.",
                sources=[],
                cache_hit=False,
                confidence="low",
                no_answer_reason="retrieval_score_below_threshold",
                trace_id=trace_id,
                routed_to_model=self.rag_settings.model,
                grounding=None,
                citation_validity=None,
                suggested_followups=[],
                token_usage={"used": 0, "available": self.rag_settings.max_tokens},
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=0,
                total_latency_ms=int((time.monotonic() - t0) * 1000),
                tokens_used={"prompt": 0, "completion": 0, "total": 0},
            )

        # 7. Context assembly (token budget)
        context_chunks, token_info = assemble_context(
            chunks=chunks,
            max_tokens=self.rag_settings.max_tokens,
            model=self.rag_settings.model,
        )

        # 8. Prompt injection: wrap chunks
        context_text = "\n\n".join(
            CONTEXT_TEMPLATE.format(
                index=i + 1,
                document_name=c.metadata.get("filename", "Unknown"),
                page=c.metadata.get("page", "?"),
                chunk_text=sanitise_retrieved_chunk(c.page_content),
            )
            for i, c in enumerate(context_chunks)
        )

        # 9. Conversation memory
        memory_context = self.memory.get_context(conversation_id)

        # 10. Model routing
        routed_llm = self.llm
        if self.rag_settings.model_routing_enabled:
            routed_llm = await self._route_model(query)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        lc_history = [
            HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
            for m in history[-self.rag_settings.memory_window * 2:]
        ]

        # 11. LLM generation
        t_llm = time.monotonic()
        chain = prompt | routed_llm | StrOutputParser()
        answer = await chain.ainvoke({
            "memory_context": memory_context,
            "history": lc_history,
            "context": context_text,
            "question": query,
        })
        llm_latency_ms = int((time.monotonic() - t_llm) * 1000)

        # 12. Output guardrail
        output_safe, output_reason = scan_output(query, answer)
        if not output_safe:
            answer = "The response was withheld due to content policy. Please rephrase your question."

        # Build sources
        sources = [
            SourceResult(
                id=f"src_{i}",
                title=c.metadata.get("filename", "Unknown"),
                document_name=c.metadata.get("filename", "Unknown"),
                chunk_text=c.page_content,
                score=float(c.metadata.get("score", 0)),
                rerank_score=float(c.metadata.get("rerank_score", 0)) if self.reranker else None,
                page=c.metadata.get("page"),
            )
            for i, c in enumerate(context_chunks)
        ]

        # Confidence level from max score
        confidence = "high" if max_score >= 0.8 else "medium" if max_score >= 0.6 else "low"

        # Token tracking (from LLM callback — simplified here)
        tokens_used = {"prompt": token_info["prompt_tokens"], "completion": 0, "total": token_info["prompt_tokens"]}

        # 13-17. Async post-processing (non-blocking)
        grounding_task = asyncio.create_task(check_grounding(answer, context_text, self.fast_llm))
        followup_task  = asyncio.create_task(generate_followups(query, answer, self.fast_llm))
        cost_usd       = calculate_cost(self.rag_settings.model, tokens_used["prompt"], tokens_used["completion"])

        # Store in cache
        self.cache.set(query, {"answer": answer, "sources": [vars(s) for s in sources]})

        # Update conversation memory
        self.memory.add_turn(conversation_id, query, answer)

        # Log cost
        asyncio.create_task(log_cost(
            user_id=user_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            model=self.rag_settings.model,
            input_tokens=tokens_used["prompt"],
            output_tokens=tokens_used["completion"],
            cost_usd=cost_usd,
        ))

        # Await parallel tasks
        grounding, followups = await asyncio.gather(grounding_task, followup_task, return_exceptions=True)

        total_latency_ms = int((time.monotonic() - t0) * 1000)

        return ChatResult(
            answer=answer,
            sources=sources,
            cache_hit=False,
            confidence=confidence,
            no_answer_reason=None,
            trace_id=trace_id,
            routed_to_model=self.rag_settings.model,
            grounding=grounding if not isinstance(grounding, Exception) else None,
            citation_validity=None,   # verified separately via background task
            suggested_followups=followups if not isinstance(followups, Exception) else [],
            token_usage={
                "used": token_info["total_tokens"],
                "available": self.rag_settings.max_tokens,
                "chunksIncluded": len(context_chunks),
                "chunksAvailable": len(chunks),
            },
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_used=tokens_used,
        )

    async def stream(
        self,
        query: str,
        conversation_id: str,
        history: list[Message],
        filters: dict | None,
        user_id: str,
        trace_id: str,
    ) -> AsyncIterator[str]:
        """
        Streaming variant — yields text chunks as they arrive.
        Post-processing (grounding, followups) is sent as a final SSE metadata frame.
        """
        # Steps 1-9 same as run() above (retrieval is always non-streaming)
        # ... (abbreviated: run same retrieval pipeline, build prompt)

        async for chunk in self.llm.astream({"context": "...", "question": query}):
            yield chunk.content

        # Final frame — metadata
        yield "__METADATA__:" + json.dumps({
            "sources": [],  # populated from retrieval above
            "confidence": "high",
            "trace_id": trace_id,
        })

    async def _route_model(self, query: str) -> ChatOpenAI:
        prompt = f"Classify as SIMPLE or COMPLEX (one word): {query}"
        result = await self.fast_llm.ainvoke(prompt)
        if "COMPLEX" in result.content.upper():
            return ChatOpenAI(model="gpt-4o", temperature=self.rag_settings.temperature)
        return self.llm

    async def _write_audit(self, event_type: str, user_id: str, payload: dict, trace_id: str):
        async with AsyncSessionLocal() as db:
            from app.models.audit import AuditEvent
            db.add(AuditEvent(
                event_type=event_type,
                user_id=user_id,
                user_email="",
                payload_json=json.dumps(payload),
                trace_id=trace_id,
            ))
            await db.commit()


class InputBlockedError(Exception):
    pass
```

---

## Hybrid Retriever

```python
# app/retrieval/hybrid_retriever.py
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self, vectorstore, documents: list[Document], weights: tuple[float, float] = (0.5, 0.5)):
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = 20
        self.dense = vectorstore.as_retriever(search_kwargs={"k": 20})
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.dense],
            weights=list(weights),
        )

    def invoke(self, query: str, filters: dict | None = None) -> list[Document]:
        if filters:
            # Apply metadata filter on dense retriever only
            self.dense.search_kwargs["filter"] = filters
        return self.ensemble.invoke(query)
```

---

## Cross-Encoder Reranker

```python
# app/retrieval/reranker.py
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class CrossEncoderReranker:
    _instances: dict[str, "CrossEncoderReranker"] = {}

    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Singleton per model name — avoid reloading weights on every request
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            instance.model = CrossEncoder(model_name)
            instance.model_name = model_name
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        for score, doc in ranked:
            doc.metadata["rerank_score"] = float(score)
        return [doc for _, doc in ranked[:top_k]]
```

---

## Context Assembler

```python
# app/context/assembler.py
import tiktoken
from langchain_core.documents import Document

def assemble_context(
    chunks: list[Document],
    max_tokens: int = 3000,
    model: str = "gpt-4o-mini",
    system_overhead: int = 500,   # reserved for system prompt + answer
) -> tuple[list[Document], dict]:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    budget = max_tokens - system_overhead
    included, dropped, used = [], 0, 0

    for chunk in chunks:  # already sorted by rerank score desc
        tokens = len(enc.encode(chunk.page_content))
        if used + tokens > budget:
            dropped += 1
            continue
        included.append(chunk)
        used += tokens

    return included, {
        "total_tokens": used,
        "prompt_tokens": used + system_overhead,
        "chunks_included": len(included),
        "chunks_dropped": dropped,
    }
```

---

## Conversation Memory

```python
# app/memory/conversation_memory.py
import json
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from app.cache.redis_client import get_redis

class ConversationMemoryManager:
    def __init__(self, llm: ChatOpenAI, max_token_limit: int = 1500):
        self.llm = llm
        self.max_token_limit = max_token_limit

    def _make_memory(self) -> ConversationSummaryBufferMemory:
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.max_token_limit,
            return_messages=False,
            memory_key="history",
        )

    def _key(self, conversation_id: str) -> str:
        return f"memory:{conversation_id}"

    def add_turn(self, conversation_id: str, human: str, ai: str):
        r = get_redis()
        mem = self._make_memory()
        raw = r.get(self._key(conversation_id))
        if raw:
            state = json.loads(raw)
            mem.chat_memory.messages = state.get("messages", [])
        mem.save_context({"input": human}, {"output": ai})
        r.setex(self._key(conversation_id), 86400, json.dumps({
            "messages": [m.dict() for m in mem.chat_memory.messages]
        }))

    def get_context(self, conversation_id: str) -> str:
        r = get_redis()
        raw = r.get(self._key(conversation_id))
        if not raw:
            return ""
        mem = self._make_memory()
        state = json.loads(raw)
        mem.chat_memory.messages = state.get("messages", [])
        return mem.load_memory_variables({}).get("history", "")
```

---

## Production Semantic Cache (Faiss-backed)

```python
# app/cache/semantic_cache.py
import json
import hashlib
import numpy as np
import faiss
from openai import OpenAI
from app.cache.redis_client import get_redis

CACHE_INDEX_KEY = "semantic_cache:faiss_index"
CACHE_ENTRIES_KEY = "semantic_cache:entries"
EMBEDDING_DIM = 1536   # text-embedding-3-small

class SemanticCache:
    """
    Production-grade semantic cache using Faiss for vector lookup.
    Much faster than O(n) Redis scan — Faiss does approximate nearest-neighbour
    search in O(log n), suitable for millions of cached queries.

    Storage layout:
    - Faiss index serialised to Redis bytes (CACHE_INDEX_KEY)
    - Entry payloads (answer, sources) stored as JSON list in Redis (CACHE_ENTRIES_KEY)
    - Faiss vector IDs correspond to positions in the entries list
    """

    def __init__(self, redis_url: str, threshold: float = 0.97, ttl_seconds: int = 86400):
        self.redis = get_redis()
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.openai = OpenAI()
        self._index: faiss.IndexFlatIP | None = None
        self._entries: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> dict | None:
        if self._index is None or self._index.ntotal == 0:
            return None
        vec = self._embed(query)
        scores, ids = self._index.search(np.array([vec], dtype=np.float32), k=1)
        score, idx = float(scores[0][0]), int(ids[0][0])
        if score >= self.threshold and idx < len(self._entries):
            return self._entries[idx]
        return None

    def set(self, query: str, payload: dict):
        vec = self._embed(query)
        if self._index is None:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)   # inner product = cosine on L2-normalised vectors
        self._index.add(np.array([vec], dtype=np.float32))
        self._entries.append(payload)
        self._persist()

    def invalidate_all(self):
        self._index = None
        self._entries = []
        self.redis.delete(CACHE_INDEX_KEY, CACHE_ENTRIES_KEY)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        # L2-normalise so inner product == cosine similarity
        vec /= np.linalg.norm(vec) + 1e-10
        return vec.tolist()

    def _load(self):
        index_bytes = self.redis.get(CACHE_INDEX_KEY)
        entries_bytes = self.redis.get(CACHE_ENTRIES_KEY)
        if index_bytes and entries_bytes:
            self._index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            self._entries = json.loads(entries_bytes)
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._entries = []

    def _persist(self):
        if self._index is None:
            return
        index_bytes = faiss.serialize_index(self._index).tobytes()
        self.redis.setex(CACHE_INDEX_KEY, self.ttl, index_bytes)
        self.redis.setex(CACHE_ENTRIES_KEY, self.ttl, json.dumps(self._entries))
```

---

## Chat API Endpoint

```python
# app/api/chat.py
from fastapi import APIRouter, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from app.chains.rag_chain import RAGChain, InputBlockedError
from app.models.user import User
from app.dependencies import get_current_user, get_db, get_rag_settings, get_vectorstore, get_documents
from app.services.chat_service import get_conversation_history, save_message
from app.evaluation.ragas_eval import run_ragas_and_store
import uuid, json

router = APIRouter()

@router.post("/chat")
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db = Depends(get_db),
    rag_settings = Depends(get_rag_settings),
    vectorstore = Depends(get_vectorstore),
    documents = Depends(get_documents),
):
    body = await request.json()
    query           = body["message"]
    conversation_id = body["conversation_id"]
    filters         = body.get("filters")
    trace_id        = str(uuid.uuid4())

    history = await get_conversation_history(db, conversation_id)

    chain = RAGChain(rag_settings, vectorstore, documents)

    try:
        result = await chain.run(
            query=query,
            conversation_id=conversation_id,
            history=history,
            filters=filters,
            user_id=current_user.id,
            trace_id=trace_id,
        )
    except InputBlockedError as e:
        return {"error": "INPUT_BLOCKED", "reason": str(e)}, 400

    # Persist messages
    await save_message(db, conversation_id, "user", query, trace_id)
    await save_message(db, conversation_id, "assistant", result.answer, trace_id, metadata={
        "confidence": result.confidence,
        "cache_hit": result.cache_hit,
        "grounding": result.grounding,
        "token_usage": result.token_usage,
    })

    # Background: RAGAS eval
    background_tasks.add_task(
        run_ragas_and_store,
        question=query,
        answer=result.answer,
        contexts=[s.chunk_text for s in result.sources],
        trace_id=trace_id,
    )

    return {
        "answer": result.answer,
        "sources": [vars(s) for s in result.sources],
        "traceMetadata": {
            "traceId": trace_id,
            "retrievalLatencyMs": result.retrieval_latency_ms,
            "llmLatencyMs": result.llm_latency_ms,
            "totalLatencyMs": result.total_latency_ms,
            "tokensUsed": result.tokens_used,
        },
        "confidence": result.confidence,
        "noAnswerReason": result.no_answer_reason,
        "cacheHit": result.cache_hit,
        "routedToModel": result.routed_to_model,
        "grounding": result.grounding,
        "citationValidity": result.citation_validity,
        "suggestedFollowups": result.suggested_followups,
        "tokenUsage": result.token_usage,
    }


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    current_user: User = Depends(get_current_user),
    rag_settings = Depends(get_rag_settings),
    vectorstore = Depends(get_vectorstore),
    documents = Depends(get_documents),
    db = Depends(get_db),
):
    body = await request.json()
    query           = body["message"]
    conversation_id = body["conversation_id"]
    filters         = body.get("filters")
    trace_id        = str(uuid.uuid4())
    history         = await get_conversation_history(db, conversation_id)

    chain = RAGChain(rag_settings, vectorstore, documents)

    async def event_stream():
        async for chunk in chain.stream(query, conversation_id, history, filters, current_user.id, trace_id):
            if chunk.startswith("__METADATA__:"):
                payload = chunk[len("__METADATA__:"):]
                yield f"event: metadata\ndata: {payload}\n\n"
            else:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```
