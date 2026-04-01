"""
POST /query endpoint.
Moved verbatim from the original main.py — no logic changes in Stage 0.
"""
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vectorstore_service import get_vectorstore, get_llm, build_qa_chain

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


qa_chain_cache = None


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question over ingested documents.
    Delegates to the shared vectorstore + LLM infrastructure.
    """
    global qa_chain_cache

    if qa_chain_cache is None:
        qa_chain_cache = build_qa_chain()

    result = qa_chain_cache.invoke({"query": req.question})
    answer = result["result"]

    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            sources.append(f"[{src}] {doc.page_content[:200]}")

    return QueryResponse(answer=answer, sources=sources)
