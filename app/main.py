"""
FastAPI application factory.
Thin wrapper — endpoints are in app/api/.
Preserves existing /query and /ingest behaviour for backward compatibility.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import create_tables

app = FastAPI(title="RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount existing endpoints from app/api/
from app.api import chat, documents, costs  # noqa: E402

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(costs.router, prefix="/api", tags=["costs"])

# Stage 6 — Feedback + Evaluation routes
from app.api import feedback, golden_dataset, experiments  # noqa: E402

app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(golden_dataset.router, prefix="/api", tags=["golden_dataset"])
app.include_router(experiments.router, prefix="/api", tags=["experiments"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.use_provider,
        "llm": settings.groq_model if settings.use_provider == "groq" else settings.minimax_model,
        "embedding_model": settings.embedding_model,
        "vectorstore": settings.vectorstore_type,
    }


@app.on_event("startup")
async def on_startup():
    """Create database tables on startup (async)."""
    await create_tables()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
