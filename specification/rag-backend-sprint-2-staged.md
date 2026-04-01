# RAG Backend — Sprint 2 (ALIGNED + STAGED)

This version is aligned with BACKEND-SETUP.md structure.

---

# 🔧 STRUCTURE ALIGNMENT (MANDATORY BEFORE STAGES)

Update base project structure to include ALL Sprint 2 modules:

app/
├── api/
│   ├── chat.py
│   ├── documents.py
│   ├── settings.py
│   ├── feedback.py
│   ├── evaluation.py        # unified (NOT evaluations.py)
│   ├── experiments.py
│   ├── audit.py
│   ├── costs.py             # keep name (NOT billing/)
│   └── connectors.py
│
├── services/
│   ├── chat_service.py
│   ├── document_service.py
│   ├── settings_service.py
│   ├── vectorstore_service.py
│   ├── phoenix_service.py
│
├── retrieval/
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── query_expander.py
│   ├── hyde.py
│   └── filters.py
│
├── ingestion/
│   ├── pipeline.py
│   ├── parsers.py
│   ├── chunker.py
│   ├── ocr.py
│   └── pii.py
│
├── memory/
│   └── conversation_memory.py
│
├── cache/
│   └── semantic_cache.py
│
├── context/
│   └── assembler.py
│
├── routing/
│   └── model_router.py
│
├── quality/
│   ├── grounding.py
│   └── citations.py
│
├── safety/
│   ├── guardrails.py
│   └── prompt_injection.py
│
├── evaluation/
│   ├── ragas_eval.py
│   ├── golden_dataset.py
│   └── experiment.py
│
├── connectors/
│   └── base.py
│
├── export/
│   └── exporter.py
│
├── workers/
│   ├── celery_app.py
│   └── ingestion_worker.py
│
├── models/
│   ├── user.py
│   ├── conversation.py
│   ├── document.py
│   ├── feedback.py
│   ├── audit.py
│   ├── evaluation.py
│   ├── settings.py
│   └── cost.py

---

# 🧱 STAGE BREAKDOWN (UPDATED)

---

## ✅ STAGE 0 — Foundation Alignment (NEW)

Goal: Ensure base system supports Sprint 2 safely

### Tasks
- Add missing folders (context, routing, connectors, export)
- Standardize naming:
  - evaluation.py (not evaluations.py)
  - costs.py (not billing/)
- Extend models:
  - Document → versioning fields
  - Message → metadata_json
  - Add QueryCost model
- Add RAGSettings new fields

### Output
- Clean structure aligned with setup
- No breaking API changes

---

## ✅ STAGE 1 — Retrieval Upgrade (B2.1)

Goal: Improve answer quality

### Tasks
- Implement HybridRetriever (dense + BM25)
- Add RRF merging
- Add reranker
- Add metadata filters
- Add query expansion (optional flag)
- Add HyDE (optional flag)

### Integration
- Modify `rag_chain.py`
- Use settings flags

### Output
- Better retrieval accuracy
- Backward compatible

---

## ✅ STAGE 2 — Ingestion Pipeline (B2.2)

Goal: Production-ready document handling

### Tasks
- Multi-format parser (PDF, DOCX, HTML, CSV)
- Chunker strategies
- OCR fallback
- Move ingestion → Celery
- Add document versioning
- Add reindex endpoint

### API
- POST /api/upload → async
- GET /api/documents/{id}/status
- POST /api/documents/{id}/reindex

---

## ✅ STAGE 3 — Safety + Grounding (B2.4)

Goal: Prevent hallucination

### Tasks
- Input guardrails
- Prompt injection detection
- Confidence threshold fallback
- Grounding checker
- Citation validator

---

## ✅ STAGE 4 — Memory + Enrichment (B2.3)

Goal: Improve UX

### Tasks
- Redis conversation memory
- Inject memory into chain
- Generate follow-up questions
- Add trace_id propagation

---

## ✅ STAGE 5 — Performance Layer (B2.5)

Goal: Reduce cost + latency

### Tasks
- Semantic cache (Redis)
- Context assembler (token control)
- Model router
- Cost tracker (persist QueryCost)

---

## ✅ STAGE 6 — Feedback + Evaluation (B2.6)

Goal: Add learning loop

### Tasks
- [x] Feedback API
- [x] RAGAS evaluation async
- [x] Golden dataset
- [x] Experiment framework

### Output
- `POST/GET /api/feedback` endpoints
- `POST/GET/DELETE /api/golden` endpoints
- `POST/GET /api/experiments` + run/status endpoints
- Celery async experiment runner with RAGAS metrics
- DB tables: `feedback`, `golden_dataset`, `experiments`, `experiment_results`

---

## ✅ STAGE 7 — Governance (B2.7)

Goal: Enterprise readiness

### Tasks
- [x] Audit logging
- [x] ACL enforcement
- [x] Retention policy
- [x] User data deletion

### Output
- `audit_log` + `retention_policy` DB tables
- `require_admin()` + `require_admin_or_self()` dependencies in `app/dependencies.py`
- `POST /api/admin/audit` (list) + `GET /api/admin/audit/export` (CSV)
- `GET/PATCH /api/settings/retention`
- `POST /api/users/me/delete` (GDPR cascade soft-delete)
- `DELETE /api/admin/users/{id}` (hard delete)
- Celery daily `retention.run` task enforcing per-resource retention from DB
- `users` table created with `is_deleted`/`deleted_at` soft-delete columns

---

## 🟢 STAGE 8 — Platform Features (B2.8)

Goal: Expand product

### Tasks
- Connectors
- Export (PDF/Markdown)
- Conversation sharing
- Full-text search

---

# 🧠 KEY ALIGNMENT FIXES SUMMARY

1. Naming unified → evaluation.py, costs.py  
2. New modules added to base structure  
3. Services layer expanded properly  
4. Models extended consistently  
5. No conflict with existing FastAPI routes  
6. Compatible with Celery + Redis infra  
7. Keeps your current architecture clean for scaling  

---

# ⚠️ Important Insight (Critical)

If you DON'T align structure now:
- your repo will become fragmented after Stage 3
- agent-generated code will mismatch imports
- scaling to multi-dev will become painful

This alignment step is **not optional** for long-term maintainability
