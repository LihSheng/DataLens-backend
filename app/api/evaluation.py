"""
Evaluation API — placeholder for Stage 6.

GET  /api/evaluations       — list evaluation runs
POST /api/evaluations/run   — trigger a new evaluation run
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.api._errors import build_error

router = APIRouter(prefix="/api", tags=["evaluation"])


class EvaluationRun(BaseModel):
    traceId: str
    question: str
    lastScore: float
    threshold: float
    passed: bool
    evaluatedAt: str


class EvaluationRunListResponse(BaseModel):
    runs: list[EvaluationRun]


class RunEvaluationResponse(BaseModel):
    message: str
    traceId: str
    status: str


# ─────────────────────────────────────────────────────────
# Stub implementations — replace with real RAGAS integration
# ─────────────────────────────────────────────────────────

_placeholder_runs: list[EvaluationRun] = []


@router.get("/evaluations", response_model=EvaluationRunListResponse)
async def list_evaluations() -> EvaluationRunListResponse:
    """
    List all historical evaluation runs.
    Currently returns an empty list until Stage 6 RAGAS integration is complete.
    """
    return EvaluationRunListResponse(runs=_placeholder_runs)


@router.post("/evaluations/run", response_model=RunEvaluationResponse)
async def run_evaluation() -> RunEvaluationResponse:
    """
    Trigger a new evaluation run for the most recent trace.
    Currently a no-op stub until Stage 6 RAGAS integration is complete.
    """
    raise HTTPException(
        status_code=501,
        detail=build_error(code="EVALUATION_NOT_IMPLEMENTED", message="Evaluation not yet implemented — Stage 6 (RAGAS integration) is pending"),
    )
