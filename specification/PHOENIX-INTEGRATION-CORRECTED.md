# Phoenix RAG Observability — Integration Guide (Corrected)

> Agent instructions: implement every section in order.
> Sections marked ⚠️ CHANGED replace the previous version of this doc.
> Sections marked ✅ UNCHANGED are valid as originally written.

---

## What Arize Phoenix Provides ✅ UNCHANGED

Arize Phoenix is an open-source ML observability platform focused on LLM and RAG pipelines.
For this system it gives:

- **Trace viewer** — end-to-end trace of every RAG call: retrieval latency, LLM latency, token counts
- **Span details** — individual spans for embedding, vector search, reranking, LLM generation
- **Retrieval quality** — which chunks were retrieved, their scores, whether they were used
- **LLM evals** — hallucination detection, relevance scoring, Q&A correctness (run async)
- **Dataset / experiment tracking** — compare retrieval configs over time

---

## Architecture Overview ⚠️ CHANGED

```
Browser (React App)
   │
   │  POST /api/chat  →  { answer, sources, traceMetadata: { traceId, ... } }
   │
   │  GET /api/phoenix/*  →  (admin only, authenticated)
   │
   ▼
RAG Backend (FastAPI)
   │
   ├── POST /api/chat → wraps request in OTel span → runs RAG chain
   ├── GET  /api/phoenix/* → proxies to Phoenix REST API (httpx)
   │
   ├── RAG chain ──► OpenTelemetry spans (LangChainInstrumentor + custom)
   │                      │
   │                      ▼
   │               Arize Phoenix (port 6006, internal only)
   │                      │
   └──────────────────────┘ (backend fetches via httpx, never browser-direct)
```

**Key rule:** The browser NEVER calls Phoenix directly.
All Phoenix data flows through `/api/phoenix/*` which requires admin auth.
`VITE_PHOENIX_URL` is only used for the "Open Phoenix" external link button.

---

## Backend — Step 1: OTel Instrumentation Setup ⚠️ NEW SECTION

### 1a. Register Phoenix tracer in `app/main.py`

```python
# app/main.py
from contextlib import asynccontextmanager
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Must be called BEFORE any LangChain objects are instantiated
    tracer_provider = register(endpoint=settings.phoenix_collector_endpoint)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    await create_tables()
    yield

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)
```

### 1b. Extract the real OTel trace ID in `app/chains/rag_chain.py`

The backend previously used `str(uuid.uuid4())` as the trace_id.
This is WRONG — Phoenix uses 32-char hex OTel trace IDs.
Replace with this helper:

```python
# app/chains/rag_chain.py

from opentelemetry import trace as otel_trace

def get_otel_trace_id() -> str:
    """Read the active OTel trace ID. Falls back to UUID in non-traced contexts."""
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")   # 32-char hex, e.g. "4bf92f3577b34da6..."
    import uuid
    return str(uuid.uuid4())
```

### 1c. Wrap the entire chat request in an OTel span in `app/api/chat.py`

```python
# app/api/chat.py

from opentelemetry import trace as otel_trace
from app.chains.rag_chain import get_otel_trace_id

@router.post("/chat")
async def chat(request: Request, ...):
    body = await request.json()
    query           = body["message"]
    conversation_id = body["conversation_id"]
    filters         = body.get("filters")

    # Wrap in a root OTel span so all LangChain child spans nest under it
    tracer = otel_trace.get_tracer("rag-api")
    with tracer.start_as_current_span("rag_request") as root_span:
        root_span.set_attribute("user.id",         current_user.id)
        root_span.set_attribute("conversation.id", conversation_id)
        root_span.set_attribute("query.length",    len(query))

        # Read the REAL OTel trace ID — not a UUID
        trace_id = get_otel_trace_id()

        history = await get_conversation_history(db, conversation_id)
        chain   = RAGChain(rag_settings, vectorstore, documents)

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
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "INPUT_BLOCKED", "reason": str(e)})

    # ... rest of handler
```

### 1d. Add custom span attributes in `app/chains/rag_chain.py`

`LangChainInstrumentor` only instruments built-in LangChain calls.
Add manual spans for your custom pipeline steps:

```python
# app/chains/rag_chain.py — inside run(), after each major step

tracer = otel_trace.get_tracer("rag-chain")

# After retrieval + reranking (step 5):
with tracer.start_as_current_span("retrieval_summary") as span:
    span.set_attribute("retrieval.raw_chunks",       len(deduped))
    span.set_attribute("retrieval.reranked_chunks",  len(chunks))
    span.set_attribute("retrieval.max_score",        max_score)
    span.set_attribute("retrieval.hybrid_weight",    self.rag_settings.hybrid_weight_dense)
    span.set_attribute("retrieval.reranker_enabled", self.rag_settings.reranker_enabled)
    span.set_attribute("retrieval.filters_applied",  bool(filters))

# After context assembly (step 8):
with tracer.start_as_current_span("context_assembly") as span:
    span.set_attribute("context.chunks_included", token_info["chunks_included"])
    span.set_attribute("context.chunks_available", len(chunks))
    span.set_attribute("context.tokens_used",     token_info["total_tokens"])
    span.set_attribute("context.tokens_budget",   self.rag_settings.max_tokens)
```

### 1e. Add OTel span to `app/retrieval/reranker.py`

```python
# app/retrieval/reranker.py

from opentelemetry import trace as otel_trace

tracer = otel_trace.get_tracer("rag-reranker")

class CrossEncoderReranker:
    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        with tracer.start_as_current_span("cross_encoder_rerank") as span:
            span.set_attribute("reranker.model",      self.model_name)
            span.set_attribute("reranker.input_docs", len(documents))
            span.set_attribute("reranker.top_k",      top_k)

            pairs  = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

            for score, doc in ranked:
                doc.metadata["rerank_score"] = float(score)

            result = [doc for _, doc in ranked[:top_k]]
            span.set_attribute("reranker.output_docs", len(result))
            if result:
                span.set_attribute("reranker.top_score",
                                   float(result[0].metadata["rerank_score"]))
            return result
```

---

## Backend — Step 2: RAGAS Eval → Phoenix Annotation ⚠️ NEW SECTION

After RAGAS scores are computed, annotate the Phoenix span so scores appear in the trace viewer.

```python
# app/evaluation/ragas_eval.py — add after storing to DB

import httpx
from app.config import settings

async def run_ragas_and_store(question: str, answer: str, contexts: list[str], trace_id: str):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        scores = result.to_pandas().iloc[0].to_dict()

        # Store to DB as before ...

        # Annotate Phoenix span
        annotations = []
        label_map = {
            "faithfulness":      "Faithfulness",
            "answer_relevancy":  "Answer Relevance",
            "context_precision": "Context Precision",
        }
        for key, label in label_map.items():
            score = scores.get(key)
            if score is None:
                continue
            annotations.append({
                "span_id": trace_id,
                "name":    label,
                "annotator_kind": "LLM",
                "result": {
                    "label":       "PASS" if float(score) >= 0.7 else "FAIL",
                    "score":       float(score),
                    "explanation": f"{label}: {float(score):.2f}",
                }
            })

        if annotations:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{settings.phoenix_base_url}/v1/span_annotations",
                    json={"data": annotations},
                )
    except Exception as e:
        logger.warning("RAGAS annotation failed for trace %s: %s", trace_id, e)
```

---

## Backend — Step 3: Feedback → Phoenix Annotation ⚠️ NEW SECTION

```python
# app/api/feedback.py — update submit_feedback

import httpx
from app.config import settings

@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, db=Depends(get_db), user=Depends(get_current_user)):
    db.add(MessageFeedback(**body.dict(), user_id=user.id))
    await db.commit()

    if body.trace_id:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{settings.phoenix_base_url}/v1/span_annotations",
                    json={"data": [{
                        "span_id": body.trace_id,
                        "name":    "Human Feedback",
                        "annotator_kind": "HUMAN",
                        "result": {
                            "label":       body.rating,
                            "score":       1.0 if body.rating == "positive" else 0.0,
                            "explanation": body.comment or "",
                        }
                    }]},
                )
        except Exception as e:
            logger.warning("Phoenix feedback annotation failed: %s", e)

    return {"status": "ok"}
```

---

## Backend — Step 4: Phoenix Proxy Routes ⚠️ NEW SECTION

Create `app/services/phoenix_service.py`:

```python
# app/services/phoenix_service.py

import httpx
from app.config import settings

async def proxy_get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{settings.phoenix_base_url}{path}",
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()
```

Create `app/api/phoenix.py`:

```python
# app/api/phoenix.py

from fastapi import APIRouter, Depends, Query
from app.dependencies import require_admin
from app.services.phoenix_service import proxy_get
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/phoenix/traces")
async def get_traces(
    limit: int = Query(default=50, le=200),
    start_time: str | None = None,
    _=Depends(require_admin),
):
    params = {"limit": limit}
    if start_time:
        params["start_time"] = start_time
    return await proxy_get("/v1/traces", params)

@router.get("/phoenix/traces/{trace_id}")
async def get_trace(trace_id: str, _=Depends(require_admin)):
    return await proxy_get(f"/v1/traces/{trace_id}")

@router.get("/phoenix/spans")
async def get_spans(trace_id: str = Query(...), _=Depends(require_admin)):
    return await proxy_get("/v1/spans", {"trace_id": trace_id})

@router.get("/phoenix/evaluations")
async def get_evaluations(trace_id: str = Query(...), _=Depends(require_admin)):
    return await proxy_get("/v1/evaluations", {"trace_id": trace_id})

@router.get("/phoenix/summary")
async def get_summary(_=Depends(require_admin)):
    start = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"
    data = await proxy_get("/v1/traces", {"limit": 1000, "start_time": start})
    traces = data.get("data", [])
    latencies = [t.get("latency_ms", 0) for t in traces if t.get("latency_ms")]
    errors = sum(1 for t in traces if t.get("status") == "ERROR")
    total  = len(traces)
    return {
        "totalQueries": total,
        "avgLatencyMs": int(sum(latencies) / len(latencies)) if latencies else 0,
        "errorRate":    round(errors / total, 3) if total else 0,
        "period":       "24h",
    }
```

Register in `app/main.py`:

```python
from app.api import phoenix as phoenix_router
app.include_router(phoenix_router.router, prefix="/api", tags=["phoenix"])
```

Add to `app/config.py`:

```python
phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
phoenix_base_url: str = "http://localhost:6006"
```

---

## Backend — Step 5: ChatResponse traceMetadata fields ⚠️ CHANGED

Add `retrieved_chunks` and `used_chunks` to `ChatResult` in `app/chains/rag_chain.py`:

```python
@dataclass
class ChatResult:
    answer: str
    sources: list[SourceResult]
    trace_id: str
    confidence: str
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
    retrieved_chunks: int    # ← ADD: total chunks before reranking
    used_chunks: int         # ← ADD: chunks that made it into context window
```

Populate in `run()`:
```python
retrieved_chunks = len(deduped)   # after dedup, before reranking
used_chunks      = len(context_chunks)  # after context assembly
```

Return in the API response:
```python
"traceMetadata": {
    "traceId":          trace_id,
    "retrievalLatencyMs": result.retrieval_latency_ms,
    "llmLatencyMs":     result.llm_latency_ms,
    "totalLatencyMs":   result.total_latency_ms,
    "tokensUsed":       result.tokens_used,
    "retrievedChunks":  result.retrieved_chunks,   # ← ADD
    "usedChunks":       result.used_chunks,         # ← ADD
},
```

---

## Frontend — Step 6: TypeScript Types ⚠️ CHANGED

```ts
// src/types/phoenix.ts  — REPLACE entire file

export interface TraceMetadata {
  traceId:           string
  retrievalLatencyMs: number
  llmLatencyMs:      number
  totalLatencyMs:    number
  tokensUsed: {
    prompt:     number
    completion: number
    total:      number
  }
  retrievedChunks: number   // total retrieved before reranking
  usedChunks:      number   // chunks included in context window
}

export interface PhoenixTrace {
  traceId:      string
  rootSpanName: string
  startTime:    string
  endTime:      string
  durationMs:   number
  status:       'OK' | 'ERROR'
  spanCount:    number
}

export interface PhoenixSpan {
  spanId:       string
  traceId:      string
  parentSpanId: string | null
  name:         string
  spanKind:     'RETRIEVER' | 'LLM' | 'CHAIN' | 'TOOL'
  startTime:    string
  endTime:      string
  durationMs:   number
  status:       'OK' | 'ERROR'
  attributes:   Record<string, unknown>
}

export interface PhoenixEval {
  name:        string
  result:      'PASS' | 'FAIL' | 'UNKNOWN'
  score:       number | null
  explanation: string
}

export interface PhoenixSummary {
  totalQueries: number
  avgLatencyMs: number
  errorRate:    number
  period:       string
}
```

---

## Frontend — Step 7: Phoenix API Service ⚠️ CHANGED

```ts
// src/services/api/phoenix.ts  — REPLACE entire file
// Uses the authenticated httpClient (not direct Phoenix URL)

import { httpClient } from '../httpClient'
import type { PhoenixTrace, PhoenixSpan, PhoenixEval, PhoenixSummary } from '@/types/phoenix'

export const phoenixApi = {
  getRecentTraces: (limit = 50) =>
    httpClient.get<{ data: PhoenixTrace[] }>('/api/phoenix/traces', { params: { limit } })
              .then(r => r.data),

  getTrace: (traceId: string) =>
    httpClient.get<PhoenixTrace>(`/api/phoenix/traces/${traceId}`)
              .then(r => r.data),

  getSpans: (traceId: string) =>
    httpClient.get<{ data: PhoenixSpan[] }>('/api/phoenix/spans', { params: { traceId } })
              .then(r => r.data),

  getEvals: (traceId: string) =>
    httpClient.get<{ data: PhoenixEval[] }>('/api/phoenix/evaluations', { params: { traceId } })
              .then(r => r.data),

  getSummary: () =>
    httpClient.get<PhoenixSummary>('/api/phoenix/summary')
              .then(r => r.data),
}
```

---

## Frontend — Step 8: `config.ts` ⚠️ CHANGED

`VITE_PHOENIX_URL` is now only used for the external "Open Phoenix" button.
All API calls go through the proxy.

```ts
// src/lib/config.ts — update to clarify scope

export const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL as string,
  appEnv:     import.meta.env.VITE_APP_ENV      as string,
  sentryDsn:  import.meta.env.VITE_SENTRY_DSN   as string,
  // Only used for the "Open Phoenix" external link — NOT for API calls
  phoenixUrl: import.meta.env.VITE_PHOENIX_URL  as string | undefined,
}
```

---

## Frontend — Step 9: `TraceLink` Component ⚠️ CHANGED

```ts
// src/features/chat/components/TraceLink.tsx  — REPLACE

import { config } from '@/lib/config'
import { ExternalLink } from 'lucide-react'

interface TraceLinkProps {
  traceId: string
}

export function TraceLink({ traceId }: TraceLinkProps) {
  // Only render if VITE_PHOENIX_URL is configured
  if (!config.phoenixUrl) return null

  // Phoenix trace viewer URL format: /traces?traceId=<hex128>
  const url = `${config.phoenixUrl}/traces?traceId=${traceId}`

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="trace-link"
      title="View full trace in Phoenix"
    >
      <ExternalLink size={12} />
      View trace
    </a>
  )
}
```

---

## Frontend — Step 10: `ChatMessage` component ✅ UNCHANGED LOGIC, updated field names

```ts
function ChatMessage({ message }: { message: Message }) {
  const can = useAuthStore(s => s.can)
  const { traceMetadata } = message

  return (
    <div className="message message--assistant">
      <MessageText text={message.text} sources={message.sources} />

      {traceMetadata && (
        <div className="message-meta">
          <LatencyBadge ms={traceMetadata.totalLatencyMs} />
          <TokenBadge tokens={traceMetadata.tokensUsed.total} />
          {can('settings:read') && (
            <TraceLink traceId={traceMetadata.traceId} />
          )}
        </div>
      )}
    </div>
  )
}
```

---

## Frontend — Step 11: Span normaliser utility ⚠️ NEW FILE

Phoenix's raw API response uses a different shape than our `PhoenixSpan` type.
This normaliser converts before passing to `SpanWaterfall`.

```ts
// src/features/observability/utils/normaliseSpans.ts  — NEW FILE

import type { PhoenixSpan } from '@/types/phoenix'

export function normaliseSpans(rawSpans: unknown[]): PhoenixSpan[] {
  return (rawSpans as any[]).map(s => ({
    spanId:       s.span_id        ?? s.spanId        ?? '',
    traceId:      s.trace_id       ?? s.traceId       ?? '',
    parentSpanId: s.parent_span_id ?? s.parentSpanId  ?? null,
    name:         s.name           ?? s.span_name     ?? 'unknown',
    spanKind:     normaliseKind(s.span_kind ?? s.spanKind),
    startTime:    s.start_time     ?? s.startTime     ?? '',
    endTime:      s.end_time       ?? s.endTime       ?? '',
    durationMs:   s.latency_ms     ?? s.durationMs    ?? 0,
    status:       (s.status_code === 'OK' || s.status === 'OK') ? 'OK' : 'ERROR',
    attributes:   s.attributes ?? {},
  }))
}

function normaliseKind(raw: string | undefined): PhoenixSpan['spanKind'] {
  const map: Record<string, PhoenixSpan['spanKind']> = {
    RETRIEVER: 'RETRIEVER', retriever: 'RETRIEVER',
    LLM: 'LLM',             llm: 'LLM',
    CHAIN: 'CHAIN',         chain: 'CHAIN',
    TOOL: 'TOOL',           tool: 'TOOL',
  }
  return map[raw ?? ''] ?? 'CHAIN'
}
```

Use in `TraceDetailDrawer`:
```ts
import { normaliseSpans } from '../utils/normaliseSpans'

const { data } = useQuery(['spans', traceId], () => phoenixApi.getSpans(traceId))
const spans = normaliseSpans(data?.data ?? [])
// Pass normalised spans to SpanWaterfall
```

---

## Frontend — Step 12: `SpanWaterfall` component ✅ UNCHANGED

No changes needed to the component itself.
Ensure it receives already-normalised spans from Step 11.

---

## Frontend — Step 13: `EvalResults` component ✅ UNCHANGED

No changes needed.

---

## Frontend — Step 14: Observability Page layout ⚠️ ADD summary endpoint

```
/observability
├── Header: "RAG Observability" + "Open Phoenix" button (uses VITE_PHOENIX_URL)
├── Summary cards — useQuery(['phoenix-summary'], phoenixApi.getSummary)
│     Total queries | Avg latency | Error rate | Period: last 24h
├── Trace list — useQuery(['phoenix-traces'], () => phoenixApi.getRecentTraces(50))
│     polling every 30s
│     Columns: Time | Latency | Tokens | Status | Evals | "View" button
└── TraceDetailDrawer (on "View"):
      ├── Query text + Answer preview
      ├── SpanWaterfall (spans from phoenixApi.getSpans → normaliseSpans)
      ├── RetrievedChunkList
      └── EvalResults (evals from phoenixApi.getEvals)
```

---

## Frontend — Step 15: MSW Handlers ⚠️ CHANGED

```ts
// src/mocks/handlers/phoenix.ts  — REPLACE

import { http, HttpResponse } from 'msw'

const MOCK_TRACES = [
  {
    traceId:      'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4',
    rootSpanName: 'rag_request',
    startTime:    new Date(Date.now() - 120000).toISOString(),
    endTime:      new Date(Date.now() - 118200).toISOString(),
    durationMs:   1800,
    status:       'OK',
    spanCount:    6,
  },
  {
    traceId:      'b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5',
    rootSpanName: 'rag_request',
    startTime:    new Date(Date.now() - 240000).toISOString(),
    endTime:      new Date(Date.now() - 236500).toISOString(),
    durationMs:   3500,
    status:       'ERROR',
    spanCount:    4,
  },
]

const MOCK_SPANS = [
  { spanId: 'span_001', traceId: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4', parentSpanId: null,       name: 'rag_request',          spanKind: 'CHAIN',    startTime: new Date(Date.now() - 120000).toISOString(), endTime: new Date(Date.now() - 118200).toISOString(), durationMs: 1800, status: 'OK', attributes: {} },
  { spanId: 'span_002', traceId: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4', parentSpanId: 'span_001', name: 'hybrid_retrieval',      spanKind: 'RETRIEVER', startTime: new Date(Date.now() - 120000).toISOString(), endTime: new Date(Date.now() - 119600).toISOString(), durationMs: 400,  status: 'OK', attributes: { 'retrieval.raw_chunks': 20, 'retrieval.max_score': 0.87 } },
  { spanId: 'span_003', traceId: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4', parentSpanId: 'span_001', name: 'cross_encoder_rerank', spanKind: 'CHAIN',    startTime: new Date(Date.now() - 119600).toISOString(), endTime: new Date(Date.now() - 119300).toISOString(), durationMs: 300,  status: 'OK', attributes: { 'reranker.input_docs': 20, 'reranker.output_docs': 5 } },
  { spanId: 'span_004', traceId: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4', parentSpanId: 'span_001', name: 'ChatOpenAI',            spanKind: 'LLM',      startTime: new Date(Date.now() - 119300).toISOString(), endTime: new Date(Date.now() - 118200).toISOString(), durationMs: 1100, status: 'OK', attributes: { 'llm.model_name': 'gpt-4o-mini', 'llm.token_count.prompt': 1240, 'llm.token_count.completion': 187 } },
]

const MOCK_EVALS: Record<string, object[]> = {
  'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4': [
    { name: 'Faithfulness',      result: 'PASS', score: 0.91, explanation: 'All claims supported by retrieved context.' },
    { name: 'Answer Relevance',  result: 'PASS', score: 0.88, explanation: 'Answer directly addresses the question.' },
    { name: 'Context Precision', result: 'PASS', score: 0.76, explanation: 'Most retrieved chunks were used.' },
    { name: 'Human Feedback',    result: 'PASS', score: 1.0,  explanation: 'positive' },
  ],
}

export const phoenixHandlers = [
  // All paths are /api/phoenix/* — proxied through FastAPI, never direct to Phoenix
  http.get('/api/phoenix/traces',           () => HttpResponse.json({ data: MOCK_TRACES })),
  http.get('/api/phoenix/traces/:traceId',  ({ params }) => {
    const t = MOCK_TRACES.find(x => x.traceId === params.traceId)
    return t ? HttpResponse.json(t) : new HttpResponse(null, { status: 404 })
  }),
  http.get('/api/phoenix/spans',            ({ request }) => {
    const traceId = new URL(request.url).searchParams.get('traceId')
    return HttpResponse.json({ data: MOCK_SPANS.filter(s => s.traceId === traceId) })
  }),
  http.get('/api/phoenix/evaluations',      ({ request }) => {
    const traceId = new URL(request.url).searchParams.get('traceId')
    return HttpResponse.json({ data: MOCK_EVALS[traceId as string] ?? [] })
  }),
  http.get('/api/phoenix/summary',          () => HttpResponse.json({
    totalQueries: 142, avgLatencyMs: 1840, errorRate: 0.021, period: '24h',
  })),
]
```

---

## Environment Variables ⚠️ CHANGED

### Backend `.env`
```
# Phoenix
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces
PHOENIX_BASE_URL=http://phoenix:6006
```

### Frontend `.env.local`
```
# Only for the external "Open Phoenix" button — optional
# Leave empty if Phoenix is not publicly accessible
VITE_PHOENIX_URL=http://localhost:6006
```

### Vercel dashboard
Do NOT set `VITE_PHOENIX_URL` in production unless Phoenix has a public HTTPS URL
behind authentication. Leave it unset — `TraceLink` will simply not render.

---

## Deployment — Phoenix in Production ✅ UNCHANGED STRUCTURE, one addition

### Option A: Self-hosted (recommended)

```yaml
# docker-compose.yml
services:
  rag-api:
    environment:
      PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces
      PHOENIX_BASE_URL: http://phoenix:6006
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"   # internal only — do not expose to internet
    volumes:
      - phoenix_data:/data
volumes:
  phoenix_data:
```

Phoenix is reachable from `rag-api` container via `http://phoenix:6006`.
The browser never reaches Phoenix directly — all access goes through `/api/phoenix/*`.

### Option B: Arize Cloud

```python
# app/services/phoenix_service.py — update proxy_get for cloud auth

async def proxy_get(path: str, params: dict | None = None) -> dict:
    headers = {}
    if settings.phoenix_api_key:
        headers["Authorization"] = f"Bearer {settings.phoenix_api_key}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{settings.phoenix_base_url}{path}",
            params=params or {},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()
```

Add to `.env`:
```
PHOENIX_BASE_URL=https://app.phoenix.arize.com
PHOENIX_API_KEY=your_arize_api_key
```

---

## Phased Rollout ✅ UNCHANGED

| Phase | Sprint | What to implement |
|---|---|---|
| 1 | Sprint 2 | Steps 1–5: OTel instrumentation, real traceId, proxy routes, RAGAS + feedback annotations |
| 2 | Sprint 3 | Steps 6–15: FE Phoenix service, TraceLink, SpanWaterfall, ObservabilityPage |
| 3 | Sprint 4 | Polling for async eval results; CSV trace export; manual re-eval trigger |

---

## Implementation Checklist

```
Backend
[ ] Step 1a — LangChainInstrumentor in main.py lifespan
[ ] Step 1b — get_otel_trace_id() helper in rag_chain.py
[ ] Step 1c — wrap chat request in OTel span in api/chat.py
[ ] Step 1d — custom span attributes in rag_chain.py run()
[ ] Step 1e — OTel span in reranker.py
[ ] Step 2  — RAGAS → Phoenix annotation in ragas_eval.py
[ ] Step 3  — Feedback → Phoenix annotation in api/feedback.py
[ ] Step 4  — phoenix_service.py + api/phoenix.py + register in main.py
[ ] Step 5  — retrieved_chunks + used_chunks in ChatResult + API response

Frontend
[ ] Step 6  — Updated types in src/types/phoenix.ts
[ ] Step 7  — Replace phoenixApi service (proxy, not direct)
[ ] Step 8  — Update config.ts (phoenixUrl scope)
[ ] Step 9  — Replace TraceLink (correct URL format)
[ ] Step 11 — Add normaliseSpans utility
[ ] Step 12 — Wire normaliseSpans in TraceDetailDrawer
[ ] Step 15 — Replace MSW handlers (/api/phoenix/* paths, {data:[]} shape)

Verification
[ ] Send a chat message; check traceId in response is 32 hex chars (no dashes)
[ ] Open http://localhost:6006 → search that traceId → trace must appear
[ ] GET /api/phoenix/traces → returns real Phoenix data (not 404)
[ ] TraceLink on a message opens Phoenix at the correct trace
[ ] SpanWaterfall shows retrieval + rerank + LLM spans
[ ] EvalResults shows RAGAS scores after ~5s background eval
[ ] Thumbs-up on a message → Phoenix trace shows "Human Feedback: positive"
```
