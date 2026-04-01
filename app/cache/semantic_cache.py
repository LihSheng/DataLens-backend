"""Redis-backed semantic cache for query/answer reuse."""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis

from app.config import settings
from app.services.vectorstore_service import embeddings

logger = logging.getLogger(__name__)

_CACHE_ENTRY_KEY = "datalens:semantic_cache:entry:{namespace}:{entry_id}"
_CACHE_INDEX_KEY = "datalens:semantic_cache:index:{namespace}"


@dataclass
class SemanticCacheHit:
    """Cache hit payload."""

    answer: str
    sources: List[Dict[str, Any]]
    similarity: float
    model: str = ""


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """
    Embedding-similarity cache using Redis as storage.

    Entries are searched by comparing the incoming query embedding
    against recent cached query embeddings in the same namespace.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        max_entries_per_namespace: int = 200,
        scan_limit: int = 50,
        similarity_threshold: float = 0.9,
    ):
        self.redis_url = redis_url or settings.redis_url
        self.ttl_seconds = ttl_seconds
        self.max_entries_per_namespace = max_entries_per_namespace
        self.scan_limit = scan_limit
        self.similarity_threshold = similarity_threshold
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
        return self._client

    @staticmethod
    def _entry_key(namespace: str, entry_id: str) -> str:
        return _CACHE_ENTRY_KEY.format(namespace=namespace, entry_id=entry_id)

    @staticmethod
    def _index_key(namespace: str) -> str:
        return _CACHE_INDEX_KEY.format(namespace=namespace)

    @staticmethod
    def _embed(text: str) -> List[float]:
        emb = embeddings.embed_query(text)
        return [float(x) for x in emb]

    def get(self, query: str, namespace: str = "global") -> Optional[SemanticCacheHit]:
        """Return the best semantic match if above threshold."""
        try:
            query_emb = self._embed(query)
            index_key = self._index_key(namespace)
            entry_ids = self.client.lrange(index_key, 0, max(0, self.scan_limit - 1))
            if not entry_ids:
                return None

            pipe = self.client.pipeline()
            for entry_id in entry_ids:
                pipe.hgetall(self._entry_key(namespace, entry_id))
            raw_entries = pipe.execute()

            best: Optional[SemanticCacheHit] = None
            for raw in raw_entries:
                if not raw:
                    continue
                created_at = float(raw.get("created_at_ts", "0"))
                if time.time() - created_at > self.ttl_seconds:
                    continue

                raw_emb = raw.get("query_embedding_json")
                if not raw_emb:
                    continue
                candidate_emb = json.loads(raw_emb)
                similarity = _cosine_similarity(query_emb, candidate_emb)
                if similarity < self.similarity_threshold:
                    continue

                hit = SemanticCacheHit(
                    answer=raw.get("answer", ""),
                    sources=json.loads(raw.get("sources_json", "[]")),
                    similarity=similarity,
                    model=raw.get("model", ""),
                )
                if best is None or hit.similarity > best.similarity:
                    best = hit

            return best
        except Exception as exc:
            logger.warning("Semantic cache read failed: %s", exc)
            return None

    def set(
        self,
        *,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        namespace: str = "global",
        model: str = "",
    ) -> None:
        """Write a new semantic cache entry."""
        try:
            query_emb = self._embed(query)
            entry_id = str(uuid.uuid4())
            now = time.time()

            key = self._entry_key(namespace, entry_id)
            index_key = self._index_key(namespace)

            payload = {
                "query": query,
                "answer": answer,
                "model": model,
                "sources_json": json.dumps(sources),
                "query_embedding_json": json.dumps(query_emb),
                "created_at_ts": str(now),
            }

            pipe = self.client.pipeline()
            pipe.hset(key, mapping=payload)
            pipe.expire(key, self.ttl_seconds)
            pipe.lpush(index_key, entry_id)
            pipe.ltrim(index_key, 0, max(0, self.max_entries_per_namespace - 1))
            pipe.expire(index_key, self.ttl_seconds)
            pipe.execute()
        except Exception as exc:
            logger.warning("Semantic cache write failed: %s", exc)
