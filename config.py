import os
from dotenv import load_dotenv
load_dotenv()

# Use Groq for now (free, fast) — swap to MiniMax when key is provided
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

# Provider selection
USE_PROVIDER = "groq"  # "groq" or "minimax"

# Embedding model (using OpenAI compat - works with MiniMax or OpenAI)
EMBEDDING_MODEL = "text-embedding-02"
OPENAI_API_KEY = MINIMAX_API_KEY or GROQ_API_KEY
OPENAI_API_BASE = "https://api.minimax.chat/v1" if USE_PROVIDER == "minimax" else "https://api.openai.com/v1"

# LLM models
GROQ_MODEL = "llama-3.3-70b-versatile"
MINIMAX_MODEL = "MiniMax-Text-01"

# Vector store
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "memory")  # memory | milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_docs")
