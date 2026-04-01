"""
Full env-based settings for RAG backend.
Preserves all existing config values (GROQ/MiniMax) + adds new fields.
"""
import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseSettings):
    # --- Existing GROQ/MiniMax provider config ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    minimax_api_key: str = os.getenv("MINIMAX_API_KEY", "")
    use_provider: str = os.getenv("USE_PROVIDER", "groq")  # "groq" or "minimax"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-02")
    openai_api_key: str = ""
    openai_api_base: str = "https://api.minimax.chat/v1"
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    minimax_model: str = os.getenv("MINIMAX_MODEL", "MiniMax-Text-01")

    # --- Vector store ---
    vectorstore_type: str = os.getenv("VECTORSTORE_TYPE", "memory")  # memory | milvus | chroma
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "rag_docs")
    chroma_persist_path: str = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma")

    # --- New Sprint 2 fields (stubs, no-op until stages implement them) ---
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb"
    )

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # App
    app_env: str = os.getenv("APP_ENV", "development")
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    allowed_origins: str = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,https://yourdomain.vercel.app"
    )

    # JWT
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

    # Celery
    celery_broker_url: str = os.getenv(
        "CELERY_BROKER_URL", "redis://localhost:6379/1"
    )
    celery_result_backend: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/2"
    )

    # Phoenix / observability
    phoenix_collector_endpoint: str = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
    )

    # File storage
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    # Reranker (optional)
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # RAG settings (new model fields)
    query_expansion_enabled: bool = _env_bool("QUERY_EXPANSION_ENABLED", False)

    # Chunking defaults
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    hyde_enabled: bool = _env_bool("HYDE_ENABLED", False)
    reranker_enabled: bool = _env_bool("RERANKER_ENABLED", False)
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

    # Stage 5
    semantic_cache_enabled: bool = _env_bool("SEMANTIC_CACHE_ENABLED", True)
    semantic_cache_threshold: float = float(
        os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.9")
    )
    context_max_tokens: int = int(os.getenv("CONTEXT_MAX_TOKENS", "1800"))
    routing_mode: str = os.getenv("ROUTING_MODE", "balanced")
    fast_model: str = os.getenv("FAST_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    quality_model: str = os.getenv("QUALITY_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Backward-compatibility aliases for existing code that imports from config
GROQ_API_KEY = settings.groq_api_key
MINIMAX_API_KEY = settings.minimax_api_key
USE_PROVIDER = settings.use_provider
EMBEDDING_MODEL = settings.embedding_model
GROQ_MODEL = settings.groq_model
MINIMAX_MODEL = settings.minimax_model
OPENAI_API_KEY = settings.openai_api_key or settings.groq_api_key
OPENAI_API_BASE = (
    "https://api.minimax.chat/v1"
    if settings.use_provider == "minimax"
    else "https://api.openai.com/v1"
)
VECTORSTORE_TYPE = settings.vectorstore_type
MILVUS_HOST = settings.milvus_host
MILVUS_PORT = settings.milvus_port
MILVUS_COLLECTION = settings.milvus_collection
