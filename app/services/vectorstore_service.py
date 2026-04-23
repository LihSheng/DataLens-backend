"""
Vector store + embedding service.
Re-exports the existing vectorstore / embedding logic from the original main.py.
No functional changes — pure migration in Stage 0.
"""
import logging
from typing import List, Optional

import faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import Milvus, FAISS, Chroma
from langchain_core.documents import Document

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

# --- Embeddings (shared singleton) ---
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- Global state (same as original main.py) ---
_vectorstore: Optional[object] = None
# _qa_chain: Optional[object] = None  # TODO: disabled - depends on langchain_classic


def get_vectorstore():
    """Returns the existing global vectorstore, initialising on first call."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if VECTORSTORE_TYPE == "milvus":
        try:
            _vectorstore = Milvus(
                embedding_function=embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name=MILVUS_COLLECTION,
            )
            return _vectorstore
        except Exception as e:
            print(f"Milvus connection failed: {e}, falling back to in-memory")

    if VECTORSTORE_TYPE == "chroma":
        # Chroma with persistent storage + BM25 hybrid (start empty; do not seed junk docs)
        _vectorstore = Chroma(
            collection_name="datalens",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_PATH,
        )
        _vectorstore._bm25_texts = []
        _vectorstore._bm25_metadata = []
        _vectorstore._bm25 = None
        return _vectorstore

    # In-memory FAISS (start empty; no placeholder docs)
    dim = len(embeddings.embed_query("dimension probe"))
    index = faiss.IndexFlatL2(dim)
    _vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )
    # Initialise BM25 index storage (populated by add_documents_with_bm25)
    _vectorstore._bm25_texts = []
    _vectorstore._bm25_metadata = []
    _vectorstore._bm25 = None
    return _vectorstore


def get_llm(model_name: Optional[str] = None, temperature: float = 0.7):
    """Returns the configured LLM (Groq or MiniMax/OpenAI-compatible)."""
    if USE_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name or GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name or MINIMAX_MODEL,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            temperature=temperature,
        )


# def build_qa_chain():
#     """Builds (or returns cached) the RetrievalQA chain."""
#     global _qa_chain
#     vs = get_vectorstore()
#     llm = get_llm()
#
#     _qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vs.as_retriever(search_kwargs={"k": 4}),
#     )
#     return _qa_chain


def add_documents_with_bm25(
    chunks: List[Document],
    texts: List[str] = None,
    metadatas: List[dict] = None,
) -> None:
    """
    Add documents to the vector store and the in-memory BM25 index.

    This enables hybrid retrieval (dense + sparse) after documents are ingested.

    Args:
        chunks: List of LangChain Document objects to add.
        texts: Optional explicit list of text strings (uses chunk.page_content if None).
        metadatas: Optional explicit list of metadata dicts (uses chunk.metadata if None).
    """
    global _vectorstore

    vs = get_vectorstore()

    # Initialise BM25 lists on first call
    if not hasattr(vs, "_bm25_texts"):
        vs._bm25_texts = []
        vs._bm25_metadata = []
        vs._bm25 = None

    # Resolve texts / metadatas
    texts = texts or [c.page_content for c in chunks]
    metadatas = metadatas or [c.metadata for c in chunks]

    # Add to vector store (Chroma uses add_texts, FAISS uses add_documents)
    if VECTORSTORE_TYPE == "chroma":
        vs.add_texts(texts=texts, metadatas=metadatas)
        vs.persist()
    else:
        vs.add_documents(chunks)

    # Extend BM25 index
    vs._bm25_texts.extend(texts)
    vs._bm25_metadata.extend(metadatas)

    # Rebuild BM25 index
    try:
        import rank_bm25
    except ImportError:
        logging.warning(
            "rank_bm25 not installed; BM25 index will not be rebuilt. "
            "Install with: pip install rank-bm25"
        )
        return

    tokenized = [t.split(" ") for t in vs._bm25_texts]
    vs._bm25 = rank_bm25.BM25Okapi(tokenized)
    logging.info(
        f"BM25 index rebuilt: {len(vs._bm25_texts)} documents indexed."
    )
