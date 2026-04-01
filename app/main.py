"""
FastAPI app entrypoint for MVP FE/BE integration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.mvp import router as mvp_router
from app.config import settings
from app.db.session import create_tables
from app.models import user as _user_model  # noqa: F401
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
app.include_router(mvp_router, prefix="/api", tags=["mvp"])


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
