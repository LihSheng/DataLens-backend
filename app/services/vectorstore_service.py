"""
Vector store + embedding service.
Re-exports the existing vectorstore / embedding logic from the original main.py.
No functional changes — pure migration in Stage 0.
"""
from typing import Optional

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus, FAISS
from langchain_classic.chains import RetrievalQA

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
)

# --- Embeddings (shared singleton) ---
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- Global state (same as original main.py) ---
_vectorstore: Optional[object] = None
_qa_chain: Optional[object] = None


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

    # In-memory FAISS
    _vectorstore = FAISS.from_texts(["initial placeholder"], embeddings)
    return _vectorstore


def get_llm():
    """Returns the configured LLM (Groq or MiniMax/OpenAI-compatible)."""
    if USE_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.7,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=MINIMAX_MODEL,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            temperature=0.7,
        )


def build_qa_chain():
    """Builds (or returns cached) the RetrievalQA chain."""
    global _qa_chain
    vs = get_vectorstore()
    llm = get_llm()

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
    )
    return _qa_chain
