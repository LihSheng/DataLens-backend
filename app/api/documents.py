"""
POST /ingest endpoint.
Moved verbatim from the original main.py — no logic changes in Stage 0.
"""
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import VECTORSTORE_TYPE, MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
from app.services.vectorstore_service import get_vectorstore, embeddings

router = APIRouter()


@router.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """
    Ingest documents — supports .txt, .md, .pdf.
    Delegates parsing and chunking to the ingestion stubs.
    """
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
    vs = get_vectorstore()

    if VECTORSTORE_TYPE == "milvus":
        try:
            from langchain_community.vectorstores import Milvus

            Milvus.from_documents(
                documents=chunks,
                embedding=embeddings,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                collection_name=MILVUS_COLLECTION,
            )
            return {
                "message": f"Ingested {len(chunks)} chunks into Milvus",
                "file_count": len(files),
                "chunks": len(chunks),
            }
        except Exception as e:
            raise HTTPException(500, f"Milvus ingest failed: {e}")
    else:
        vs.add_documents(chunks)
        return {
            "message": f"Ingested {len(chunks)} chunks into memory",
            "file_count": len(files),
            "chunks": len(chunks),
        }
