# Backend Setup — Day 1 Bootstrapping Guide

> FastAPI + LangChain + PostgreSQL + Redis + Celery + Docker
> Companion to: RAG-TODO.md, ARCHITECTURE.md

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app factory
│   ├── config.py                # Settings from env vars
│   ├── dependencies.py          # Shared FastAPI deps (db, auth, settings)
│   │
│   ├── api/                     # Route handlers (thin — delegate to services)
│   │   ├── auth.py
│   │   ├── chat.py
│   │   ├── documents.py
│   │   ├── settings.py
│   │   ├── feedback.py
│   │   ├── users.py
│   │   ├── evaluation.py
│   │   ├── audit.py
│   │   └── costs.py
│   │
│   ├── models/                  # SQLAlchemy ORM models + Pydantic schemas
│   │   ├── user.py
│   │   ├── conversation.py
│   │   ├── message.py
│   │   ├── document.py
│   │   ├── feedback.py
│   │   ├── audit.py
│   │   ├── evaluation.py
│   │   └── settings.py
│   │
│   ├── services/                # Business logic layer
│   │   ├── auth_service.py
│   │   ├── chat_service.py
│   │   ├── document_service.py
│   │   └── settings_service.py
│   │
│   ├── chains/
│   │   └── rag_chain.py         # Main LangChain RAG chain (see RAG-CHAIN.md)
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── query_expander.py
│   │
│   ├── ingestion/
│   │   ├── pipeline.py          # Orchestrates parse → chunk → embed → store
│   │   ├── parsers.py
│   │   ├── chunker.py
│   │   └── ocr.py
│   │
│   ├── memory/
│   │   └── conversation_memory.py
│   │
│   ├── cache/
│   │   └── semantic_cache.py
│   │
│   ├── safety/
│   │   ├── guardrails.py
│   │   └── prompt_injection.py
│   │
│   ├── quality/
│   │   ├── grounding.py
│   │   └── citations.py
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py
│   │   └── golden_dataset.py
│   │
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── ingestion_worker.py
│   │
│   └── db/
│       ├── session.py           # SQLAlchemy session factory
│       └── migrations/          # Alembic migration files
│
├── tests/
│   ├── conftest.py
│   ├── test_chat.py
│   ├── test_retrieval.py
│   └── test_ingestion.py
│
├── scripts/
│   └── eval_gate.py
│
├── docker-compose.yml
├── docker-compose.prod.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── alembic.ini
└── .env.example
```

---

## Environment Variables

```bash
# .env.example

# FastAPI
APP_ENV=development
SECRET_KEY=change-me-in-production-min-32-chars
ALLOWED_ORIGINS=http://localhost:5173,https://yourdomain.vercel.app

# Database
DATABASE_URL=postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb

# Redis
REDIS_URL=redis://localhost:6379/0

# Vector store (choose one)
VECTOR_STORE=chroma                          # chroma | pinecone | weaviate
CHROMA_PERSIST_PATH=./data/chroma
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_INDEX_NAME=rag-index

# LLM
OPENAI_API_KEY=
DEFAULT_LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Reranker
COHERE_API_KEY=                              # optional, for Cohere Rerank
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Phoenix observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces

# File storage
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50

# JWT
JWT_SECRET=change-me-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: str = "development"
    secret_key: str
    allowed_origins: list[str] = ["http://localhost:5173"]

    database_url: str
    redis_url: str

    vector_store: str = "chroma"
    chroma_persist_path: str = "./data/chroma"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-index"

    openai_api_key: str
    default_llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    cohere_api_key: str = ""
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"

    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## FastAPI App Factory

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from opentelemetry import trace
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from app.config import settings
from app.api import auth, chat, documents, settings as settings_router
from app.api import feedback, users, evaluation, audit, costs
from app.db.session import create_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Instrument LangChain → Phoenix on startup
    tracer_provider = register(endpoint=settings.phoenix_collector_endpoint)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    await create_tables()
    yield

app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,       prefix="/api/auth",      tags=["auth"])
app.include_router(chat.router,       prefix="/api",           tags=["chat"])
app.include_router(documents.router,  prefix="/api",           tags=["documents"])
app.include_router(settings_router.router, prefix="/api",      tags=["settings"])
app.include_router(feedback.router,   prefix="/api",           tags=["feedback"])
app.include_router(users.router,      prefix="/api",           tags=["users"])
app.include_router(evaluation.router, prefix="/api",           tags=["evaluation"])
app.include_router(audit.router,      prefix="/api",           tags=["audit"])
app.include_router(costs.router,      prefix="/api",           tags=["costs"])
```

---

## Database Schema (PostgreSQL + SQLAlchemy)

```python
# app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

```python
# app/models/user.py
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    id:             Mapped[str]      = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email:          Mapped[str]      = mapped_column(String, unique=True, nullable=False)
    name:           Mapped[str]      = mapped_column(String, nullable=False)
    hashed_password: Mapped[str]     = mapped_column(String, nullable=False)
    role:           Mapped[str]      = mapped_column(String, default="user")   # 'admin' | 'user'
    is_active:      Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:     Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login_at:  Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

```python
# app/models/conversation.py
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id:         Mapped[str]      = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:    Mapped[str]      = mapped_column(ForeignKey("users.id"), nullable=False)
    title:      Mapped[str]      = mapped_column(String, default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages:   Mapped[list]     = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id:              Mapped[str]      = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str]      = mapped_column(ForeignKey("conversations.id"), nullable=False)
    role:            Mapped[str]      = mapped_column(String)   # 'user' | 'assistant'
    content:         Mapped[str]      = mapped_column(String)
    trace_id:        Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json:   Mapped[str | None] = mapped_column(String, nullable=True)  # JSON blob for confidence, grounding etc
    created_at:      Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation:    Mapped["Conversation"] = relationship(back_populates="messages")
```

```python
# app/models/document.py
import uuid
from datetime import datetime
from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base

class Document(Base):
    __tablename__ = "documents"

    id:                  Mapped[str]       = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:             Mapped[str]       = mapped_column(ForeignKey("users.id"), nullable=False)
    name:                Mapped[str]       = mapped_column(String, nullable=False)
    file_path:           Mapped[str]       = mapped_column(String, nullable=False)
    size:                Mapped[int]       = mapped_column(Integer, default=0)
    extension:           Mapped[str]       = mapped_column(String)
    status:              Mapped[str]       = mapped_column(String, default="processing")
    parse_error:         Mapped[str | None] = mapped_column(String, nullable=True)
    ocr_applied:         Mapped[bool]      = mapped_column(Boolean, default=False)
    pii_entities_found:  Mapped[str | None] = mapped_column(String, nullable=True)  # JSON list
    chunking_strategy:   Mapped[str]       = mapped_column(String, default="recursive")
    version:             Mapped[int]       = mapped_column(Integer, default=1)
    parent_document_id:  Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    is_active_version:   Mapped[bool]      = mapped_column(Boolean, default=True)
    created_at:          Mapped[datetime]  = mapped_column(DateTime, default=datetime.utcnow)

class DocumentAcl(Base):
    __tablename__ = "document_acl"

    id:              Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id:     Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    principal_type:  Mapped[str] = mapped_column(String)   # 'user' | 'role'
    principal_id:    Mapped[str] = mapped_column(String)
    can_read:        Mapped[bool] = mapped_column(Boolean, default=True)
```

```python
# app/models/audit.py
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base

class AuditEvent(Base):
    __tablename__ = "audit_events"

    id:           Mapped[str]      = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type:   Mapped[str]      = mapped_column(String)
    user_id:      Mapped[str]      = mapped_column(String)
    user_email:   Mapped[str]      = mapped_column(String)
    ip_address:   Mapped[str | None] = mapped_column(String, nullable=True)
    payload_json: Mapped[str | None] = mapped_column(String, nullable=True)
    trace_id:     Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class QueryCost(Base):
    __tablename__ = "query_costs"

    id:              Mapped[str]   = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:         Mapped[str]   = mapped_column(String)
    conversation_id: Mapped[str]   = mapped_column(String)
    trace_id:        Mapped[str]   = mapped_column(String)
    model:           Mapped[str]   = mapped_column(String)
    input_tokens:    Mapped[int]   = mapped_column(Integer, default=0)
    output_tokens:   Mapped[int]   = mapped_column(Integer, default=0)
    cost_usd:        Mapped[float] = mapped_column(default=0.0)
    timestamp:       Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class GoldenQuestion(Base):
    __tablename__ = "golden_questions"

    id:                   Mapped[str]   = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question:             Mapped[str]   = mapped_column(String)
    expected_answer:      Mapped[str]   = mapped_column(String)
    relevant_document_ids: Mapped[str | None] = mapped_column(String, nullable=True)  # JSON list
    min_faithfulness:     Mapped[float] = mapped_column(default=0.8)
    min_relevance:        Mapped[float] = mapped_column(default=0.75)
    last_faithfulness:    Mapped[float | None] = mapped_column(nullable=True)
    last_relevance:       Mapped[float | None] = mapped_column(nullable=True)
    last_run_at:          Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at:           Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class MessageFeedback(Base):
    __tablename__ = "message_feedback"

    id:              Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id:      Mapped[str] = mapped_column(String)
    conversation_id: Mapped[str] = mapped_column(String)
    trace_id:        Mapped[str | None] = mapped_column(String, nullable=True)
    rating:          Mapped[str] = mapped_column(String)   # 'positive' | 'negative'
    comment:         Mapped[str | None] = mapped_column(String, nullable=True)
    created_at:      Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

---

## Alembic Setup

```bash
pip install alembic
alembic init app/db/migrations
```

```python
# alembic.ini — set sqlalchemy.url
sqlalchemy.url = postgresql+psycopg2://raguser:ragpass@localhost:5432/ragdb
```

```python
# app/db/migrations/env.py — target_metadata
from app.db.session import Base
from app.models import user, conversation, document, audit   # import all models
target_metadata = Base.metadata
```

```bash
# Generate and run first migration
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

---

## Redis Setup

```python
# app/cache/redis_client.py
import redis.asyncio as redis
from app.config import settings

_redis: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis
```

Uses:
- `redis://localhost:6379/0` — general cache + semantic cache
- `redis://localhost:6379/1` — Celery broker
- `redis://localhost:6379/2` — Celery results
- `redis://localhost:6379/3` — conversation memory buffers

---

## Celery Worker

```python
# app/workers/celery_app.py
from celery import Celery
from app.config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.ingestion_worker"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # fair scheduling for long ingestion tasks
)
```

```python
# app/workers/ingestion_worker.py
from app.workers.celery_app import celery_app
from app.ingestion.pipeline import run_ingestion
from app.db.session import AsyncSessionLocal
from app.models.document import Document
import asyncio

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_document(self, document_id: str, file_path: str, settings_dict: dict):
    try:
        asyncio.run(_run(document_id, file_path, settings_dict))
    except Exception as exc:
        raise self.retry(exc=exc)

async def _run(document_id, file_path, settings_dict):
    async with AsyncSessionLocal() as db:
        try:
            await run_ingestion(document_id, file_path, settings_dict, db)
            doc = await db.get(Document, document_id)
            doc.status = "ready"
            await db.commit()
        except Exception as e:
            doc = await db.get(Document, document_id)
            doc.status = "failed"
            doc.parse_error = str(e)
            await db.commit()
            raise
```

---

## Auth Dependency

```python
# app/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.config import settings
from app.db.session import get_db
from app.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    return user

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
```

---

## Docker Compose (Development)

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./data:/app/data
    env_file: .env
    depends_on:
      - postgres
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build: .
    env_file: .env
    depends_on:
      - postgres
      - redis
    command: celery -A app.workers.celery_app worker --loglevel=info --concurrency=2

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
    volumes:
      - phoenix_data:/data

volumes:
  postgres_data:
  redis_data:
  phoenix_data:
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies for OCR and PDF parsing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/uploads /app/data/chroma

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## requirements.txt

```
# Core
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.0
pydantic-settings==2.2.1
python-multipart==0.0.9

# Database
sqlalchemy[asyncio]==2.0.29
asyncpg==0.29.0
psycopg2-binary==2.9.9
alembic==1.13.1

# Auth
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Redis + Celery
redis==5.0.3
celery==5.3.6

# LangChain
langchain==0.2.0
langchain-openai==0.1.6
langchain-community==0.2.0
langchain-experimental==0.0.59
langchain-core==0.2.0
openai==1.25.0

# Vector store
chromadb==0.5.0
pinecone-client==3.2.2    # optional

# Retrieval
rank-bm25==0.2.2
sentence-transformers==2.7.0

# Ingestion
pypdf==4.2.0
python-docx==1.1.0
beautifulsoup4==4.12.3
unstructured[pdf,docx]==0.13.5

# OCR
pytesseract==0.3.10
pdf2image==1.17.0

# PII
presidio-analyzer==2.2.354
presidio-anonymizer==2.2.354

# Safety
llm-guard==0.3.6

# Evaluation
ragas==0.1.9
datasets==2.19.0

# Observability
arize-phoenix==4.0.0
opentelemetry-sdk==1.24.0
openinference-instrumentation-langchain==0.1.19

# Tokenisation
tiktoken==0.7.0

# Export
weasyprint==62.1
```

---

## First-Run Checklist

```bash
# 1. Copy env
cp .env.example .env
# Fill in OPENAI_API_KEY, JWT_SECRET, SECRET_KEY

# 2. Start services
docker-compose up -d postgres redis phoenix

# 3. Install Python deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # for PII detection

# 4. Run migrations
alembic upgrade head

# 5. Start API
uvicorn app.main:app --reload

# 6. Start Celery worker (separate terminal)
celery -A app.workers.celery_app worker --loglevel=info

# 7. Verify
curl http://localhost:8000/docs        # FastAPI Swagger UI
open http://localhost:6006             # Phoenix dashboard
```
