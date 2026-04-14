"""
FastAPI app entrypoint for MVP FE/BE integration.
"""
import logging
logger = logging.getLogger(__name__)
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
from app.config import settings
from app.db.session import create_tables

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

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(admin_users_router, prefix="/api", tags=["admin-users"])
app.include_router(mvp_router, prefix="/api", tags=["mvp"])
app.include_router(phoenix_proxy_router, prefix="/api", tags=["phoenix"])
app.include_router(audit_router, prefix="/api", tags=["admin"])
app.include_router(feedback_router, prefix="/api", tags=["feedback"])
app.include_router(evaluation_router, tags=["evaluation"])
app.include_router(costs_router, tags=["costs"])


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


@app.on_event("startup")
async def on_startup():
    await create_tables()

    # Register OTel tracer with Phoenix collector
    resource = Resource(attributes={"service.name": "rag-backend"})
    provider = TracerProvider(resource=resource)
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as e:
        logger.warning(f"OTel: Phoenix exporter unavailable, traces will not be exported. Error: {e}")
    from opentelemetry.trace import set_tracer_provider
    set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
