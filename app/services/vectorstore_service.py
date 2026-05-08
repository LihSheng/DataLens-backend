"""
DEPRECATED — use app.core.vectorstore and app.core.llm_provider instead.

This module remains as a backward-compatibility shim. All new code should
inject adapters via app.state or factory functions from app.core.
"""
import logging

logger = logging.getLogger(__name__)

from app.core.vectorstore import (
    ChromaVectorStore,
    FAISSVectorStore,
    IndexedChunk,
    MilvusVectorStore,
    VectorStore,
    create_vectorstore,
)
from app.core.llm_provider import (
    CircuitOpenError,
    GroqProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    create_llm_provider,
)

# Legacy globals — lazy-initialised once, forwarding to the new modules.

_legacy_vs: object = None
_legacy_llm: object = None
_legacy_embeddings: object = None

from app.config import (
    USE_PROVIDER,
    GROQ_API_KEY,
    MINIMAX_API_KEY,
    OPENAI_API_BASE,
    GROQ_MODEL,
    MINIMAX_MODEL,
    VECTORSTORE_TYPE,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    CHROMA_PERSIST_PATH,
)

# --- Embeddings (shared singleton, backward compat) ---

embeddings = None  # will be set on first access


def _ensure_embeddings():
    global embeddings, _legacy_embeddings
    if _legacy_embeddings is None:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        _legacy_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        embeddings = _legacy_embeddings
    return _legacy_embeddings


_ensure_embeddings()


def get_vectorstore():
    logger.warning(
        "get_vectorstore() is deprecated; inject a VectorStore adapter instead. "
        "See app.core.vectorstore."
    )
    global _legacy_vs
    if _legacy_vs is not None:
        return _legacy_vs

    from app.core.vectorstore import create_vectorstore as _create_vs

    _legacy_vs = _create_vs(
        backend=VECTORSTORE_TYPE,
        chroma_persist_path=CHROMA_PERSIST_PATH,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
        milvus_collection=MILVUS_COLLECTION,
    )
    return _legacy_vs


def get_llm(model_name=None, temperature=0.7, timeout=60.0):
    logger.warning(
        "get_llm() is deprecated; inject an LLMProvider adapter instead. "
        "See app.core.llm_provider."
    )
    global _legacy_llm
    if _legacy_llm is None:
        from app.core.llm_provider import create_llm_provider as _create_llm

        _legacy_llm = _create_llm(
            provider=USE_PROVIDER,
            groq_api_key=GROQ_API_KEY,
            groq_model=GROQ_MODEL,
            minimax_api_key=MINIMAX_API_KEY,
            minimax_model=MINIMAX_MODEL,
        )
    return _legacy_llm.get_llm(model=model_name, temperature=temperature, timeout=timeout)


def add_documents_with_bm25(chunks, texts=None, metadatas=None):
    logger.warning(
        "add_documents_with_bm25() is deprecated; use vectorstore.add() instead. "
        "See app.core.vectorstore."
    )
    from app.core.vectorstore import IndexedChunk as _Chunk

    vs = get_vectorstore()
    if texts is None:
        texts = [c.page_content for c in chunks]
    if metadatas is None:
        metadatas = [c.metadata for c in chunks]
    indexed = [_Chunk(content=t, metadata=m) for t, m in zip(texts, metadatas)]
    vs.add(indexed)
