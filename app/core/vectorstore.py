"""
VectorStore — deep module for hybrid (dense + BM25) document indexing and retrieval.

Hides FAISS/Chroma/Milvus backend selection, BM25 lifecycle, and RRF fusion
behind a single interface with four methods: search, search_by_vector, add, delete.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class IndexedChunk:
    content: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 8) -> List[Document]:
        """Hybrid search: dense + BM25 fused via Reciprocal Rank Fusion."""

    @abstractmethod
    def search_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Document]:
        """Dense-only vector similarity search — HyDE needs this."""

    @abstractmethod
    def add(self, chunks: Sequence[IndexedChunk]) -> int:
        """Atomically index chunks into dense + BM25. Returns count indexed."""

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> int:
        """Remove from both indexes. Returns count removed."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Used by semantic cache."""

    @abstractmethod
    def is_ready(self) -> bool:
        """For health checks."""


# ---------------------------------------------------------------------------
# Embedding helper (shared across adapters)
# ---------------------------------------------------------------------------

_embeddings: Optional[object] = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        _embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# BM25 internals
# ---------------------------------------------------------------------------


class _BM25Index:
    """Internal sparse index — only used by FAISS and Chroma adapters."""

    def __init__(self) -> None:
        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self._bm25: Optional[object] = None
        self._import_warned = False

    def add(self, texts: List[str], metadatas: List[dict]) -> None:
        self.texts.extend(texts)
        self.metadata.extend(metadatas)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            import rank_bm25
        except ImportError:
            if not self._import_warned:
                logger.warning("rank_bm25 not installed; BM25 disabled")
                self._import_warned = True
            return

        if self.texts:
            tokenized = [t.split(" ") for t in self.texts]
            self._bm25 = rank_bm25.BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> List[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.split(" "))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            Document(page_content=self.texts[i], metadata=self.metadata[i])
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (static — pure function, no I/O)
# ---------------------------------------------------------------------------


def _rrf_fusion(
    dense_results: List[tuple],
    sparse_results: List[tuple],
    rrf_k: int = 60,
) -> List[Document]:
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _score) in enumerate(dense_results):
        doc_id = doc.page_content[:50] + str(doc.metadata)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc

    for rank, (doc, _score) in enumerate(sparse_results):
        doc_id = doc.page_content[:50] + str(doc.metadata)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


# ---------------------------------------------------------------------------
# FAISS adapter
# ---------------------------------------------------------------------------


class FAISSVectorStore(VectorStore):
    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._vs: Optional[object] = None
        self._bm25 = _BM25Index()
        self._persist_path = persist_path
        self._dim: Optional[int] = None

    def _ensure_init(self) -> object:
        if self._vs is not None:
            return self._vs

        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        emb = _get_embeddings()
        self._dim = len(emb.embed_query("dimension probe"))
        index = faiss.IndexFlatL2(self._dim)
        self._vs = FAISS(
            embedding_function=emb,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        return self._vs

    def search(self, query: str, k: int = 8) -> List[Document]:
        vs = self._ensure_init()
        dense = vs.similarity_search_with_score(query, k=max(k, 20))
        sparse = [(d, 1.0 / (i + 1)) for i, d in enumerate(self._bm25.search(query, k=max(k, 20)))]
        fused = _rrf_fusion(dense, sparse)
        return fused[:k]

    def search_by_vector(self, embedding: List[float], k: int = 4) -> List[Document]:
        vs = self._ensure_init()
        return vs.similarity_search_by_vector(embedding, k=k)

    def add(self, chunks: Sequence[IndexedChunk]) -> int:
        vs = self._ensure_init()
        docs = [Document(page_content=c.content, metadata=c.metadata) for c in chunks]
        vs.add_documents(docs)
        self._bm25.add(
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.debug("FAISS index: %d chunks added", len(chunks))
        return len(chunks)

    def delete(self, chunk_ids: List[str]) -> int:
        vs = self._ensure_init()
        if hasattr(vs, "delete"):
            return vs.delete(chunk_ids)
        return 0

    def embed_query(self, text: str) -> List[float]:
        emb = _get_embeddings()
        return [float(x) for x in emb.embed_query(text)]

    def is_ready(self) -> bool:
        try:
            self._ensure_init()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Chroma adapter
# ---------------------------------------------------------------------------


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_path: str = "./data/chroma") -> None:
        self._persist_path = persist_path
        self._vs: Optional[object] = None
        self._bm25 = _BM25Index()

    def _ensure_init(self) -> object:
        if self._vs is not None:
            return self._vs

        from langchain_community.vectorstores import Chroma

        emb = _get_embeddings()
        self._vs = Chroma(
            collection_name="datalens",
            embedding_function=emb,
            persist_directory=self._persist_path,
        )
        return self._vs

    def search(self, query: str, k: int = 8) -> List[Document]:
        vs = self._ensure_init()
        dense = vs.similarity_search_with_score(query, k=max(k, 20))
        sparse = [(d, 1.0 / (i + 1)) for i, d in enumerate(self._bm25.search(query, k=max(k, 20)))]
        fused = _rrf_fusion(dense, sparse)
        return fused[:k]

    def search_by_vector(self, embedding: List[float], k: int = 4) -> List[Document]:
        vs = self._ensure_init()
        return vs.similarity_search_by_vector(embedding, k=k)

    def add(self, chunks: Sequence[IndexedChunk]) -> int:
        vs = self._ensure_init()
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = None

        # Try Chroma 0.6+ collection.upsert or 0.4/0.5 add_texts
        add_func = getattr(vs, "add_texts", None)
        if add_func is not None:
            add_func(texts=texts, metadatas=metadatas)
        else:
            try:
                vs._collection.add(metadatas=metadatas, documents=texts, ids=ids or [str(i) for i in range(len(texts))])
            except Exception:
                docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
                vs.add_documents(docs)

        self._bm25.add(texts=texts, metadatas=metadatas)
        logger.debug("Chroma index: %d chunks added", len(chunks))
        return len(chunks)

    def delete(self, chunk_ids: List[str]) -> int:
        vs = self._ensure_init()
        delete_func = getattr(vs, "delete", None)
        if delete_func is not None:
            return delete_func(chunk_ids)
        return 0

    def embed_query(self, text: str) -> List[float]:
        emb = _get_embeddings()
        return [float(x) for x in emb.embed_query(text)]

    def is_ready(self) -> bool:
        try:
            self._ensure_init()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Milvus adapter (dense-only, no BM25)
# ---------------------------------------------------------------------------


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection: str = "rag_docs",
    ) -> None:
        self._host = host
        self._port = port
        self._collection = collection
        self._vs: Optional[object] = None

    def _ensure_init(self) -> object:
        if self._vs is not None:
            return self._vs

        from langchain_community.vectorstores import Milvus

        emb = _get_embeddings()
        self._vs = Milvus(
            embedding_function=emb,
            connection_args={"host": self._host, "port": self._port},
            collection_name=self._collection,
        )
        return self._vs

    def search(self, query: str, k: int = 8) -> List[Document]:
        vs = self._ensure_init()
        return vs.similarity_search(query, k=k)

    def search_by_vector(self, embedding: List[float], k: int = 4) -> List[Document]:
        vs = self._ensure_init()
        return vs.similarity_search_by_vector(embedding, k=k)

    def add(self, chunks: Sequence[IndexedChunk]) -> int:
        vs = self._ensure_init()
        docs = [Document(page_content=c.content, metadata=c.metadata) for c in chunks]
        vs.add_documents(docs)
        return len(chunks)

    def delete(self, chunk_ids: List[str]) -> int:
        vs = self._ensure_init()
        delete_func = getattr(vs, "delete", None)
        if delete_func is not None:
            return delete_func(chunk_ids)
        return 0

    def embed_query(self, text: str) -> List[float]:
        emb = _get_embeddings()
        return [float(x) for x in emb.embed_query(text)]

    def is_ready(self) -> bool:
        try:
            self._ensure_init()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vectorstore(
    backend: str = "chroma",
    chroma_persist_path: str = "./data/chroma",
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
    milvus_collection: str = "rag_docs",
    faiss_persist_path: Optional[str] = None,
) -> VectorStore:
    if backend == "milvus":
        try:
            return MilvusVectorStore(
                host=milvus_host,
                port=milvus_port,
                collection=milvus_collection,
            )
        except Exception as e:
            logger.warning("Milvus connection failed, falling back to FAISS: %s", e)
            return FAISSVectorStore(persist_path=faiss_persist_path)

    if backend == "faiss":
        return FAISSVectorStore(persist_path=faiss_persist_path)

    return ChromaVectorStore(persist_path=chroma_persist_path)
