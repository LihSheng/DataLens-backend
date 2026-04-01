# Incremental BM25 Index — Design & Implementation

## The Core Constraint

`rank-bm25` builds its index as a full in-memory matrix of term frequencies
across the entire corpus. It has no `add_document()` method because BM25's IDF
scores are *global* — adding one document changes the IDF weight of every term
in the vocabulary. There is no mathematical shortcut; the index must be rebuilt
from the full corpus whenever a document is added or removed.

The strategy therefore is:

1. **Store the raw corpus in Redis** (cheap appends, fast bulk reads)
2. **Mark the in-memory index as dirty** when the corpus changes
3. **Rebuild lazily** — only when a retrieval request actually arrives and the
   index is dirty, not immediately on ingest
4. **Share rebuild across workers** — use a Redis-backed rebuild lock so
   multiple Celery/API workers don't all rebuild simultaneously
5. **Trigger a proactive warm-up** after each ingest completes, in the background,
   so the first real query after an upload doesn't pay the rebuild latency

---

## Redis Key Layout

```
bm25:corpus:chunks          LIST   — serialised chunk texts (one entry per chunk)
bm25:corpus:metadata        LIST   — JSON metadata per chunk (same index as chunks)
bm25:needs_rebuild          STRING — "1" if corpus changed since last index build
bm25:rebuild_lock           STRING — distributed lock (SET NX EX 60)
bm25:index_version          STRING — increments on every successful rebuild (used for cache invalidation)
bm25:doc:{doc_id}:chunk_ids LIST   — chunk indices belonging to this document (for deletion)
```

---

## Full Implementation

```python
# app/retrieval/bm25_index.py

from __future__ import annotations

import json
import logging
import time
import threading
from typing import Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.cache.redis_client import get_redis

logger = logging.getLogger(__name__)

CORPUS_CHUNKS_KEY    = "bm25:corpus:chunks"
CORPUS_METADATA_KEY  = "bm25:corpus:metadata"
NEEDS_REBUILD_KEY    = "bm25:needs_rebuild"
REBUILD_LOCK_KEY     = "bm25:rebuild_lock"
INDEX_VERSION_KEY    = "bm25:index_version"
DOC_CHUNK_IDS_PREFIX = "bm25:doc:"


def _tokenise(text: str) -> list[str]:
    """
    Shared tokeniser used for both index building and query processing.
    Lowercases and splits on whitespace/punctuation.
    Keep it simple — consistency between index-time and query-time matters
    more than sophistication.
    """
    import re
    return re.findall(r'\b\w+\b', text.lower())


class IncrementalBM25Index:
    """
    BM25 index backed by a Redis corpus store.

    Architecture:
    - Corpus (raw texts + metadata) is persisted in Redis lists.
    - The actual BM25Okapi object lives in-memory per process instance.
    - A 'dirty flag' in Redis signals that the corpus has changed and the
      in-memory index needs rebuilding.
    - A distributed lock prevents multiple workers from rebuilding simultaneously.
    - Lazy rebuild: the index is rebuilt on the first retrieval after a change,
      not on every ingest event.
    - Background warm-up: after each document ingest, a background task
      triggers a proactive rebuild so users don't pay latency on the next query.

    Thread safety: a threading.Lock protects the in-process _index attribute
    during the rebuild window.
    """

    def __init__(self, k: int = 10, rebuild_threshold_secs: float = 0.0):
        self._index: Optional[BM25Okapi] = None
        self._index_version: int = -1
        self._lock = threading.Lock()
        self.k = k
        # Minimum seconds between rebuilds even if dirty flag is set.
        # Set > 0 in high-throughput ingest scenarios to batch-coalesce rebuilds.
        self.rebuild_threshold_secs = rebuild_threshold_secs
        self._last_rebuild_at: float = 0.0

    # ------------------------------------------------------------------
    # Write path — called from the Celery ingestion worker
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, chunks: list[Document]) -> None:
        """
        Append chunks to the Redis corpus and mark the index dirty.
        Does NOT rebuild the index — that happens lazily on next retrieval.

        Args:
            doc_id:  The document's database UUID.
            chunks:  Chunked Document objects with .page_content and .metadata.
        """
        r = get_redis()
        pipe = r.pipeline(transaction=True)

        chunk_indices: list[int] = []

        for chunk in chunks:
            # Current length = 0-based index of the chunk we're about to add
            idx = r.llen(CORPUS_CHUNKS_KEY)
            chunk_indices.append(idx)

            pipe.rpush(CORPUS_CHUNKS_KEY, chunk.page_content)
            pipe.rpush(CORPUS_METADATA_KEY, json.dumps({
                **chunk.metadata,
                "doc_id": doc_id,
                "chunk_idx": idx,
            }))

        # Store which indices belong to this document (needed for deletion)
        doc_key = f"{DOC_CHUNK_IDS_PREFIX}{doc_id}:chunk_ids"
        for idx in chunk_indices:
            pipe.rpush(doc_key, idx)

        # Mark index dirty
        pipe.set(NEEDS_REBUILD_KEY, "1")

        pipe.execute()

        logger.info(
            "BM25 corpus: appended %d chunks for doc %s, index marked dirty",
            len(chunks), doc_id
        )

    def remove_document(self, doc_id: str) -> None:
        """
        Remove all chunks belonging to a document.

        Because Redis lists don't support index-based deletion, we use a
        tombstone strategy: overwrite the chunk text with an empty string.
        On next rebuild, empty chunks are filtered out.
        The chunk_ids list for this document is then deleted.

        Note: this causes the in-memory corpus to drift from the Redis corpus
        over time as tombstones accumulate. A full compaction (LRANGE + re-write)
        runs automatically during each rebuild.
        """
        r = get_redis()
        doc_key = f"{DOC_CHUNK_IDS_PREFIX}{doc_id}:chunk_ids"

        chunk_ids_raw = r.lrange(doc_key, 0, -1)
        if not chunk_ids_raw:
            logger.warning("BM25: no chunk ids found for doc %s", doc_id)
            return

        pipe = r.pipeline(transaction=True)
        for idx_bytes in chunk_ids_raw:
            idx = int(idx_bytes)
            # LSET replaces the value at position idx with a tombstone marker
            pipe.lset(CORPUS_CHUNKS_KEY, idx, "__DELETED__")
            pipe.lset(CORPUS_METADATA_KEY, idx, json.dumps({"__deleted__": True}))

        pipe.delete(doc_key)
        pipe.set(NEEDS_REBUILD_KEY, "1")
        pipe.execute()

        logger.info(
            "BM25 corpus: tombstoned %d chunks for doc %s",
            len(chunk_ids_raw), doc_id
        )

    # ------------------------------------------------------------------
    # Read path — called from the retrieval chain
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """
        Retrieve top-k chunks matching the query.
        Rebuilds the index first if the dirty flag is set.
        """
        self._ensure_fresh()

        with self._lock:
            if self._index is None:
                logger.warning("BM25 index is empty — returning no results")
                return []
            index_snapshot = self._index
            metadata_snapshot = self._metadata_snapshot

        k = top_k or self.k
        tokens = _tokenise(query)
        scores = index_snapshot.get_scores(tokens)

        # Get top-k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            meta = metadata_snapshot[idx]
            doc = Document(
                page_content=index_snapshot.corpus[idx],
                metadata={**meta, "score": float(scores[idx])},
            )
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Rebuild logic
    # ------------------------------------------------------------------

    def _ensure_fresh(self) -> None:
        """Check dirty flag; rebuild if needed."""
        r = get_redis()
        needs_rebuild = r.get(NEEDS_REBUILD_KEY)
        if not needs_rebuild:
            return

        # Rate-limit rebuilds if rebuild_threshold_secs > 0
        now = time.monotonic()
        if now - self._last_rebuild_at < self.rebuild_threshold_secs:
            return

        # Check current index version vs our cached version
        current_version = int(r.get(INDEX_VERSION_KEY) or 0)
        if current_version == self._index_version:
            # Another worker already rebuilt and incremented the version
            r.delete(NEEDS_REBUILD_KEY)
            return

        self._rebuild_with_lock(r)

    def _rebuild_with_lock(self, r) -> None:
        """
        Acquire a distributed lock before rebuilding so only one worker
        rebuilds at a time. Other workers will wait briefly then re-check
        the version number.
        """
        # Try to acquire rebuild lock (NX = only if not exists, EX = 60s TTL)
        acquired = r.set(REBUILD_LOCK_KEY, "1", nx=True, ex=60)
        if not acquired:
            # Another worker is rebuilding — wait for it and re-read version
            logger.debug("BM25: rebuild lock held by another worker, waiting...")
            for _ in range(30):   # wait up to 3 seconds
                time.sleep(0.1)
                current_version = int(r.get(INDEX_VERSION_KEY) or 0)
                if current_version != self._index_version:
                    self._load_from_redis(r, current_version)
                    return
            logger.warning("BM25: timed out waiting for rebuild lock")
            return

        try:
            t0 = time.monotonic()
            self._do_rebuild(r)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info("BM25 index rebuilt in %dms", elapsed_ms)
        finally:
            r.delete(REBUILD_LOCK_KEY)

    def _do_rebuild(self, r) -> None:
        """Load full corpus from Redis, filter tombstones, build BM25Okapi."""
        raw_chunks   = r.lrange(CORPUS_CHUNKS_KEY, 0, -1)
        raw_metadata = r.lrange(CORPUS_METADATA_KEY, 0, -1)

        corpus: list[str] = []
        metadata: list[dict] = []

        for text_bytes, meta_bytes in zip(raw_chunks, raw_metadata):
            text = text_bytes if isinstance(text_bytes, str) else text_bytes.decode()
            meta = json.loads(meta_bytes)

            # Filter tombstones and empty chunks
            if text == "__DELETED__" or meta.get("__deleted__") or not text.strip():
                continue

            corpus.append(text)
            metadata.append(meta)

        if not corpus:
            logger.warning("BM25: corpus is empty after filtering tombstones")
            with self._lock:
                self._index = None
                self._metadata_snapshot = []
            return

        # Compact: rewrite Redis lists without tombstones
        # This runs atomically under the rebuild lock so no concurrent writes
        self._compact_corpus(r, corpus, metadata)

        # Build index
        tokenised = [_tokenise(text) for text in corpus]
        new_index = BM25Okapi(tokenised)

        # Increment version and clear dirty flag atomically
        new_version = r.incr(INDEX_VERSION_KEY)
        r.delete(NEEDS_REBUILD_KEY)

        with self._lock:
            self._index = new_index
            self._index.corpus = corpus          # store for result reconstruction
            self._metadata_snapshot = metadata
            self._index_version = int(new_version)
            self._last_rebuild_at = time.monotonic()

        logger.info(
            "BM25 rebuilt: %d chunks, version=%s", len(corpus), new_version
        )

    def _compact_corpus(self, r, corpus: list[str], metadata: list[dict]) -> None:
        """
        Rewrite Redis lists without tombstones.
        Runs under the rebuild lock so safe to delete + re-write.
        Also resets all doc chunk_id mappings to match new positions.
        """
        pipe = r.pipeline(transaction=True)
        pipe.delete(CORPUS_CHUNKS_KEY)
        pipe.delete(CORPUS_METADATA_KEY)

        # Clear all existing doc → chunk_id mappings
        for key in r.scan_iter(f"{DOC_CHUNK_IDS_PREFIX}*"):
            pipe.delete(key)

        pipe.execute()

        # Batch write in groups of 500 to avoid huge pipelines
        BATCH = 500
        doc_chunk_map: dict[str, list[int]] = {}

        for i in range(0, len(corpus), BATCH):
            pipe = r.pipeline(transaction=False)
            batch_texts = corpus[i:i + BATCH]
            batch_meta  = metadata[i:i + BATCH]
            for j, (text, meta) in enumerate(zip(batch_texts, batch_meta)):
                global_idx = i + j
                # Update chunk_idx in metadata to new compacted position
                meta["chunk_idx"] = global_idx
                pipe.rpush(CORPUS_CHUNKS_KEY, text)
                pipe.rpush(CORPUS_METADATA_KEY, json.dumps(meta))
                doc_id = meta.get("doc_id")
                if doc_id:
                    doc_chunk_map.setdefault(doc_id, []).append(global_idx)
            pipe.execute()

        # Rewrite doc → chunk_id mappings
        pipe = r.pipeline(transaction=False)
        for doc_id, indices in doc_chunk_map.items():
            key = f"{DOC_CHUNK_IDS_PREFIX}{doc_id}:chunk_ids"
            for idx in indices:
                pipe.rpush(key, idx)
        pipe.execute()

    def _load_from_redis(self, r, version: int) -> None:
        """Load metadata snapshot when another worker already rebuilt."""
        raw_metadata = r.lrange(CORPUS_METADATA_KEY, 0, -1)
        metadata = [
            json.loads(m) for m in raw_metadata
            if not json.loads(m).get("__deleted__")
        ]
        # We can't reconstruct BM25Okapi from Redis without the full corpus +
        # tokenisation, so trigger a local rebuild on this worker too.
        self._do_rebuild(r)

    # ------------------------------------------------------------------
    # Stats / health
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        r = get_redis()
        return {
            "corpus_size":     r.llen(CORPUS_CHUNKS_KEY),
            "needs_rebuild":   bool(r.get(NEEDS_REBUILD_KEY)),
            "index_version":   r.get(INDEX_VERSION_KEY),
            "index_in_memory": self._index is not None,
            "index_doc_count": len(self._index.corpus) if self._index else 0,
        }
```

---

## Wiring Into `HybridRetriever`

Replace the `BM25Retriever.from_documents()` call with `IncrementalBM25Index`:

```python
# app/retrieval/hybrid_retriever.py
from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.retrieval.bm25_index import IncrementalBM25Index

# Singleton — one shared index instance per process
_bm25_index: IncrementalBM25Index | None = None

def get_bm25_index() -> IncrementalBM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = IncrementalBM25Index(k=20)
    return _bm25_index


class BM25IndexRetriever(BaseRetriever):
    """
    LangChain-compatible wrapper around IncrementalBM25Index.
    Implements the BaseRetriever interface so it can slot into EnsembleRetriever.
    """
    k: int = Field(default=20)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return get_bm25_index().retrieve(query, top_k=self.k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        # BM25 is CPU-bound; run in thread pool to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: get_bm25_index().retrieve(query, top_k=self.k)
        )


class HybridRetriever:
    def __init__(self, vectorstore, weights: tuple[float, float] = (0.5, 0.5)):
        self.bm25_retriever = BM25IndexRetriever(k=20)
        self.dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        self.weights = list(weights)

    def invoke(self, query: str, filters: dict | None = None) -> list[Document]:
        if filters:
            self.dense_retriever.search_kwargs["filter"] = filters

        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=self.weights,
        )
        return ensemble.invoke(query)
```

---

## Wiring Into the Ingestion Worker

Call `add_document()` at the end of a successful ingest, then enqueue a proactive warm-up:

```python
# app/workers/ingestion_worker.py  (updated section)
from app.retrieval.bm25_index import get_bm25_index

async def _run(document_id: str, file_path: str, settings_dict: dict):
    async with AsyncSessionLocal() as db:
        try:
            chunks = await run_ingestion(document_id, file_path, settings_dict, db)

            # 1. Update document status
            doc = await db.get(Document, document_id)
            doc.status = "ready"
            await db.commit()

            # 2. Append to BM25 corpus (fast — just Redis writes)
            get_bm25_index().add_document(document_id, chunks)

            # 3. Enqueue proactive warm-up so next query doesn't pay rebuild latency
            warmup_bm25_index.delay()

        except Exception as e:
            doc = await db.get(Document, document_id)
            doc.status = "failed"
            doc.parse_error = str(e)
            await db.commit()
            raise


@celery_app.task
def warmup_bm25_index():
    """
    Trigger a proactive rebuild in the Celery worker process.
    Runs after each successful ingest. Since it acquires the Redis rebuild lock,
    only one rebuild runs at a time across all workers.
    """
    index = get_bm25_index()
    index._ensure_fresh()
    logger.info("BM25 warm-up complete: %s", index.stats())
```

---

## Wiring Document Deletion

```python
# app/api/documents.py  (updated delete handler)
from app.retrieval.bm25_index import get_bm25_index
from app.cache.semantic_cache import SemanticCache

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db = Depends(get_db),
    user = Depends(get_current_user),
):
    doc = await db.get(Document, document_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(404)

    # 1. Remove from vectorstore
    vectorstore.delete(filter={"doc_id": document_id})

    # 2. Tombstone chunks in BM25 corpus
    get_bm25_index().remove_document(document_id)

    # 3. Invalidate semantic cache (answers may reference deleted content)
    SemanticCache().invalidate_all()

    # 4. Remove DB record
    await db.delete(doc)
    await db.commit()

    return {"status": "deleted"}
```

---

## Startup: Load Existing Corpus on Boot

On API startup, trigger a warm-up so the first request doesn't pay cold-start latency.
Add this to the `lifespan` function in `app/main.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing Phoenix / LangChain instrumentation ...

    # Warm up BM25 index from persisted Redis corpus
    import asyncio
    from app.retrieval.bm25_index import get_bm25_index
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, get_bm25_index()._ensure_fresh)

    yield
```

---

## Performance Characteristics

| Operation | Time | Notes |
|---|---|---|
| `add_document()` — 50 chunks | ~5ms | Redis pipeline writes only |
| `remove_document()` — 50 chunks | ~5ms | Tombstone writes only |
| `rebuild` — 10k chunks | ~800ms | One-time, then cached |
| `rebuild` — 100k chunks | ~8s | Run in background worker, not API process |
| `retrieve()` — warm index | ~2ms | Pure in-memory BM25 scoring |
| `retrieve()` — cold (needs rebuild) | rebuild time + 2ms | Amortised across queries |

For corpora above ~100k chunks, move rebuilds entirely to the Celery worker pool
and have API workers only call `retrieve()` against the already-warmed in-memory index.

---

## Configuration Recommendations

```python
# Development / small corpus (< 10k chunks)
IncrementalBM25Index(k=20, rebuild_threshold_secs=0)
# Rebuilds immediately on dirty flag

# Production / medium corpus (10k–100k chunks)
IncrementalBM25Index(k=20, rebuild_threshold_secs=5.0)
# Coalesces rapid consecutive uploads into a single rebuild

# High-volume ingest (> 100k chunks, batch uploads)
# Set rebuild_threshold_secs=30 and rely on warmup_bm25_index Celery task
```

---

## Testing

```python
# tests/test_bm25_index.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.retrieval.bm25_index import IncrementalBM25Index, _tokenise


def make_doc(text: str, doc_id: str = "doc_1") -> Document:
    return Document(page_content=text, metadata={"doc_id": doc_id, "filename": "test.pdf"})


class TestTokeniser:
    def test_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenise("hello, world!") == ["hello", "world"]

    def test_product_codes(self):
        # Product codes with hyphens are kept as single tokens
        tokens = _tokenise("part XR-4920 required")
        assert "xr" in tokens or "xr-4920" in tokens  # depends on regex


class TestIncrementalBM25Index:
    @pytest.fixture
    def index(self, fake_redis):
        """IncrementalBM25Index wired to a fake Redis."""
        with patch("app.retrieval.bm25_index.get_redis", return_value=fake_redis):
            yield IncrementalBM25Index(k=5)

    def test_add_and_retrieve(self, index):
        docs = [
            make_doc("The quick brown fox", "doc_1"),
            make_doc("Python is a programming language", "doc_2"),
            make_doc("FastAPI is a web framework for Python", "doc_3"),
        ]
        for doc in docs:
            index.add_document(doc.metadata["doc_id"], [doc])

        results = index.retrieve("Python web framework")
        assert len(results) > 0
        assert any("Python" in r.page_content for r in results)

    def test_remove_document_excluded_from_results(self, index):
        index.add_document("doc_1", [make_doc("secret confidential data", "doc_1")])
        index.add_document("doc_2", [make_doc("public information here", "doc_2")])

        index.remove_document("doc_1")

        results = index.retrieve("secret confidential")
        doc_ids = [r.metadata["doc_id"] for r in results]
        assert "doc_1" not in doc_ids

    def test_empty_corpus_returns_empty(self, index):
        results = index.retrieve("anything")
        assert results == []

    def test_stats(self, index):
        index.add_document("doc_1", [make_doc("test chunk", "doc_1")])
        stats = index.stats()
        assert "corpus_size" in stats
        assert stats["needs_rebuild"] is True
```
