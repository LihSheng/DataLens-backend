"""
FastAPI RAG Server
Endpoints:
  POST /ingest         - Upload documents (PDF, TXT, MD)
  POST /query          - Ask a question over ingested docs
  GET  /health         - Health check
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus, FAISS
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config import *

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vector store
vectorstore: Optional[object] = None
qa_chain: Optional[object] = None

# Embeddings — using HuggingFace BGE (free, local)
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def get_vectorstore():
    global vectorstore
    if vectorstore is not None:
        return vectorstore
    
    if VECTORSTORE_TYPE == "milvus":
        try:
            vectorstore = Milvus(
                embedding_function=embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name=MILVUS_COLLECTION,
            )
            return vectorstore
        except Exception as e:
            print(f"Milvus connection failed: {e}, falling back to in-memory")
    
    # In-memory FAISS
    vectorstore = FAISS.from_texts(["initial placeholder"], embeddings)
    return vectorstore


def get_llm():
    if USE_PROVIDER == "groq":
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.7,
        )
    else:
        return ChatOpenAI(
            model=MINIMAX_MODEL,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            temperature=0.7,
        )


def build_qa_chain():
    global qa_chain
    vs = get_vectorstore()
    llm = get_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
    )
    return qa_chain


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": USE_PROVIDER,
        "llm": GROQ_MODEL if USE_PROVIDER == "groq" else MINIMAX_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "vectorstore": VECTORSTORE_TYPE,
    }


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """Ingest documents — supports .txt, .md, .pdf"""
    docs = []
    
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".txt", ".md", ".pdf"]:
            raise HTTPException(400, f"Unsupported file type: {suffix}")
        
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
        finally:
            os.unlink(tmp_path)
    
    if not docs:
        raise HTTPException(400, "No documents loaded")
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    print(f"[RAG] Ingested {len(chunks)} chunks from {len(files)} file(s)")
    
    # Add to vector store
    global vectorstore
    vs = get_vectorstore()
    
    if VECTORSTORE_TYPE == "milvus":
        try:
            Milvus.from_documents(
                documents=chunks,
                embedding=embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name=MILVUS_COLLECTION,
            )
            return {"message": f"Ingested {len(chunks)} chunks into Milvus", "file_count": len(files), "chunks": len(chunks)}
        except Exception as e:
            raise HTTPException(500, f"Milvus ingest failed: {e}")
    else:
        vs.add_documents(chunks)
        return {"message": f"Ingested {len(chunks)} chunks into memory", "file_count": len(files), "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global qa_chain
    
    if qa_chain is None:
        build_qa_chain()
    
    result = qa_chain.invoke({"query": req.question})
    answer = result["result"]
    
    # Collect source docs if available
    sources = []
    if "source_documents" in result:
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            sources.append(f"[{src}] {doc.page_content[:200]}")
    
    return QueryResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn
    # Dedicated local dev port to reduce collisions with other services/VMs.
    port = int(__import__("os").getenv("PORT", "6333"))
    uvicorn.run(app, host="0.0.0.0", port=port)
