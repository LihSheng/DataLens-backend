"""
FastAPI app entrypoint for MVP FE/BE integration.
"""
import logging
logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.admin_users import router as admin_users_router
from app.api.mvp import router as mvp_router
from app.api.phoenix_proxy import router as phoenix_proxy_router
from app.api.audit import router as audit_router
from app.api.feedback import router as feedback_router
from app.api.evaluation import router as evaluation_router
from app.api.costs import router as costs_router
from app.api.health import router as health_router
from app.api._errors import (
    request_id_middleware,
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.config import settings
from app.core import create_vectorstore, create_llm_provider
from app.db.session import create_tables
from fastapi.exceptions import HTTPException, RequestValidationError

# OTel / Phoenix instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace as otel_trace
from openinference.instrumentation.langchain import LangChainInstrumentor
from app.models import user as _user_model  # noqa: F401
from app.models import user_block_log as _user_block_log_model  # noqa: F401
from app.models import conversation as _conversation_model  # noqa: F401
from app.models import document as _document_model  # noqa: F401
from app.models import feedback as _feedback_model  # noqa: F401
from app.models import share_token as _share_token_model  # noqa: F401
from app.models import app_setting as _app_setting_model  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()

    # Build injected adapters and store in app.state
    app.state.vectorstore = create_vectorstore(
        backend=settings.vectorstore_type,
        chroma_persist_path=settings.chroma_persist_path,
        milvus_host=settings.milvus_host,
        milvus_port=settings.milvus_port,
        milvus_collection=settings.milvus_collection,
    )
    app.state.llm_provider = create_llm_provider(
        provider=settings.use_provider,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        minimax_api_key=settings.minimax_api_key,
        minimax_model=settings.minimax_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    # Register OTel tracer with Phoenix collector
    resource = Resource(attributes={"service.name": "rag-backend"})
    provider = TracerProvider(resource=resource)
    if settings.phoenix_enabled and settings.otel_export_enabled:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception as e:
            logger.warning(
                f"OTel: exporter unavailable, traces will not be exported. Error: {e}"
            )
    from opentelemetry.trace import set_tracer_provider
    set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)

    yield


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID middleware + standardized error handlers
app.middleware("http")(request_id_middleware)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(admin_users_router, prefix="/api", tags=["admin-users"])
app.include_router(mvp_router, prefix="/api", tags=["mvp"])
app.include_router(phoenix_proxy_router, prefix="/api", tags=["phoenix"])
app.include_router(audit_router, prefix="/api", tags=["admin"])
app.include_router(feedback_router, prefix="/api", tags=["feedback"])
app.include_router(evaluation_router, tags=["evaluation"])
app.include_router(costs_router, tags=["costs"])
app.include_router(health_router, prefix="/api", tags=["health"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.use_provider,
        "llm": settings.groq_model if settings.use_provider == "groq" else settings.minimax_model,
        "embedding_model": settings.embedding_model,
        "vectorstore": settings.vectorstore_type,
        "dev_auth_bypass": settings.dev_auth_bypass,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(__import__("os").getenv("PORT", "6333"))
    uvicorn.run(app, host="0.0.0.0", port=port)
