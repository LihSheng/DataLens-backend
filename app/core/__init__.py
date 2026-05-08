"""
app.core — deep modules for VectorStore and LLMProvider.

These are the two external seams of the RAG backend. Everything else
(hybrid retrieval, BM25, reranking, etc.) lives behind these interfaces.

Usage:
    from app.core import create_vectorstore, create_llm_provider, VectorStore, LLMProvider
"""
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

__all__ = [
    "VectorStore",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "MilvusVectorStore",
    "IndexedChunk",
    "create_vectorstore",
    "LLMProvider",
    "GroqProvider",
    "OpenAICompatibleProvider",
    "CircuitOpenError",
    "create_llm_provider",
]
