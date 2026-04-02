# pheonix-integration-staged.md

> Consolidated execution plan for Phoenix Observability integration
> Sources merged:
>
> * PHOENIX-INTEGRATION.md 
> * PHOENIX-INTEGRATION-CORRECTED.md 
> * PHOENIX-WIRING-FIX.md 

---

# 🎯 Objective

Build a **production-grade observability layer** for your RAG system using Arize Phoenix with:

* End-to-end tracing (OTel)
* Retrieval + LLM visibility
* Eval scoring (RAGAS + human feedback)
* Admin observability UI
* Secure architecture (no direct FE → Phoenix)

---

# 🧠 Core Principles (Must Follow)

1. **Single Source of Truth = OTel traceId**

   * NEVER use UUID for traceId
   * Always extract from OpenTelemetry

2. **Frontend NEVER calls Phoenix directly**

   * All calls go through `/api/phoenix/*`

3. **Tracing must wrap entire request**

   * Root span = `rag_request`

4. **Observability = Data + UX**

   * Not just logs — must be explorable in UI

---

# 🧱 STAGE 0 — Infrastructure Setup

## Goal

Bring Phoenix online + connect to backend

## Tasks

* Add Phoenix to docker-compose

```yaml
phoenix:
  image: arizephoenix/phoenix:latest
  ports:
    - "6006:6006"
```

* Backend config:

```python
phoenix_collector_endpoint = "http://phoenix:6006/v1/traces"
phoenix_base_url = "http://phoenix:6006"
```

## Outcome

* Phoenix UI accessible
* Collector ready to receive traces

---

# 🔗 STAGE 1 — OTel Instrumentation (CRITICAL FOUNDATION)

## Goal

Ensure **every RAG request produces a valid trace**

## Tasks

### 1. Register tracer (app startup)

```python
tracer_provider = register(endpoint=settings.phoenix_collector_endpoint)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

---

### 2. Wrap chat endpoint

```python
with tracer.start_as_current_span("rag_request"):
```

Add attributes:

* user.id
* conversation.id
* query.length

---

### 3. Extract REAL traceId

```python
format(ctx.trace_id, "032x")
```

❌ Remove:

```python
uuid.uuid4()
```

---

## Outcome

* Every request appears in Phoenix
* TraceLink will NOT 404 anymore

---

# 🔍 STAGE 2 — Deep Observability (Custom Spans)

## Goal

Make traces **useful**, not just visible

## Tasks

### Add spans:

#### Retrieval summary

* raw_chunks
* reranked_chunks
* max_score
* filters_applied

#### Context assembly

* chunks_used
* tokens_used
* token_budget

---

### Reranker instrumentation

```python
with tracer.start_as_current_span("cross_encoder_rerank"):
```

Attributes:

* model
* input_docs
* output_docs
* top_score

---

## Outcome

Phoenix now shows:

* Retrieval quality
* Context compression
* Reranking impact

---

# 🧪 STAGE 3 — Evaluation Integration (RAGAS + Feedback)

## Goal

Turn traces into **measurable quality signals**

## Tasks

### 1. RAGAS → Phoenix

POST:

```
/v1/span_annotations
```

Metrics:

* Faithfulness
* Answer Relevance
* Context Precision

---

### 2. Feedback → Phoenix

```json
{
  "name": "Human Feedback",
  "label": "positive | negative"
}
```

---

## Outcome

Each trace now contains:

* LLM eval scores
* Human feedback
* Pass/Fail signals

---

# 🔐 STAGE 4 — Backend Phoenix Proxy (SECURITY LAYER)

## Goal

Prevent exposing Phoenix directly

## Tasks

### Create proxy service

```python
proxy_get("/v1/traces")
```

---

### API routes

```
GET /api/phoenix/traces
GET /api/phoenix/traces/{id}
GET /api/phoenix/spans
GET /api/phoenix/evaluations
GET /api/phoenix/summary
```

---

### Enforce

```python
Depends(require_admin)
```

---

## Outcome

* Secure access
* No CORS issues
* No public Phoenix exposure

---

# 📦 STAGE 5 — ChatResponse Contract Upgrade

## Goal

Expose observability to frontend

## Add fields

```ts
traceMetadata:
  traceId
  latency
  tokens
  retrievedChunks
  usedChunks
```

---

## Outcome

Frontend can:

* show latency
* show token usage
* link to trace

---

# 🎨 STAGE 6 — Frontend Integration (Phase 1)

## Goal

Minimal but powerful UX

## Tasks

### Chat UI

* Latency badge
* Token badge
* TraceLink (admin only)

---

### TraceLink

```
/traces?traceId=<otel-id>
```

---

## Outcome

* Instant visibility
* Zero complexity UI

---

# 📊 STAGE 7 — Observability Page (Phase 2)

## Goal

Internal debugging dashboard

## Route

```
/observability
```

---

## Layout

### Header

* Open Phoenix button

### Summary

* total queries
* avg latency
* error rate

### Trace List

* polling (30s)

### Detail Drawer

* span waterfall
* eval scores
* retrieved chunks

---

## Outcome

You now have:

* mini Datadog for your RAG

---

# 🧩 STAGE 8 — Span Normalisation Layer

## Goal

Fix Phoenix API inconsistency

## Add

```ts
normaliseSpans()
```

---

## Outcome

* Stable frontend rendering
* No schema mismatch issues

---

# 🧪 STAGE 9 — MSW + Mock Data

## Goal

Frontend dev without backend dependency

## Mock endpoints

```
/api/phoenix/*
```

---

## Outcome

* Faster FE iteration
* Safe testing

---

# 🚀 STAGE 10 — Deployment Strategy

## Recommended

### Backend + Phoenix (same network)

```
frontend → backend → phoenix
```

---

## Avoid

❌ Direct FE → Phoenix

---

## Optional

* Arize Cloud (if scaling later)

---

# 📈 FINAL STATE ARCHITECTURE

```
Frontend (React)
   │
   ├── /api/chat
   ├── /api/phoenix/*
   │
   ▼
Backend (FastAPI)
   │
   ├── RAG Pipeline
   ├── OTel spans
   │
   ▼
Phoenix (internal only)
```

---

# 🧠 Execution Order (STRICT)

1. Stage 0 → Infra
2. Stage 1 → OTel (must work first)
3. Stage 2 → Custom spans
4. Stage 4 → Proxy (security)
5. Stage 5 → Response contract
6. Stage 6 → UI (basic)
7. Stage 3 → Eval + feedback
8. Stage 7 → Observability page
9. Stage 8 → Normaliser
10. Stage 9 → Mocking

---

# ⚠️ Common Failure Points

* ❌ Using UUID instead of OTel traceId
* ❌ Frontend calling Phoenix directly
* ❌ Missing root span → no trace tree
* ❌ No proxy → security leak
* ❌ No span attributes → useless traces

---

# 🧩 Final Result

You will get:

* Full RAG trace visibility
* Debuggable retrieval pipeline
* Measurable LLM quality
* Admin observability dashboard
* Production-safe architecture

---

# 🔮 Next Evolution (Optional)

* Alerting (high latency / hallucination spikes)
* Auto-retraining triggers from feedback
* Cost tracking per trace
* Multi-tenant observability

---
