"""
Experiments API — Stage 6.

POST /api/experiments           — create a new experiment
GET  /api/experiments           — list experiments (paginated)
GET  /api/experiments/{id}      — get experiment details + individual results
POST /api/experiments/{id}/run  — trigger async evaluation run (Celery task)
GET  /api/experiments/{id}/status — poll current status

The experiment framework allows comparing different retrieval/chain configs
against the golden dataset using RAGAS evaluation.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.evaluation import experiment as exp_module
from app.evaluation.golden_dataset import get_all_golden_entries

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """Retrieval / chain configuration to test in an experiment."""

    # Retrieval settings
    query_expansion: Optional[bool] = False
    hyde: Optional[bool] = False
    reranker: Optional[bool] = False
    k: Optional[int] = 8
    rerank_top_n: Optional[int] = 4

    # Model settings
    model: Optional[str] = None
    temperature: Optional[float] = 0.7

    # RAGAS metrics to compute
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of RAGAS metrics: faithfulness, answer_correctness, "
                    "answer_relevance, context_precision, context_recall, context_relevancy",
    )

    # Filter options
    document_ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None

    class Config:
        extra = "allow"  # Allow additional config fields


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    config: ExperimentConfig
    created_by: Optional[str] = None


class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    status: str
    created_by: Optional[str]
    created_at: str
    completed_at: Optional[str]


class ExperimentDetailResponse(ExperimentResponse):
    results: List[Dict[str, Any]]


class ExperimentRunRequest(BaseModel):
    """Optional per-run overrides for the experiment config."""
    config_override: Optional[Dict[str, Any]] = None


class ExperimentStatusResponse(BaseModel):
    id: str
    status: str
    completed_at: Optional[str]


class ExperimentListResponse(BaseModel):
    items: List[ExperimentResponse]
    total: int
    limit: int
    offset: int


# ─────────────────────────────────────────────────────────
# Celery task import (deferred to avoid circular imports)
# ─────────────────────────────────────────────────────────

def _get_run_task():
    """Lazily import the Celery task to avoid top-level circular dependency."""
    from app.workers.celery_app import run_experiment_task
    return run_experiment_task


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@router.post("", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    payload: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new experiment record.

    The experiment is created in 'pending' status. Use POST /{id}/run to
    trigger the actual evaluation.
    """
    config_dict = payload.config.model_dump(exclude_none=False)
    exp = await exp_module.create_experiment(
        db,
        name=payload.name,
        config=config_dict,
        description=payload.description,
        created_by=payload.created_by,
    )
    return ExperimentResponse(**exp)


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    List all experiments, newest first.
    """
    experiments, total = await exp_module.list_experiments(db, limit=limit, offset=offset)
    return ExperimentListResponse(
        items=[ExperimentResponse(**e) for e in experiments],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get experiment details including all individual evaluation results.
    """
    exp = await exp_module.get_experiment(db, experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    return ExperimentDetailResponse(**exp)


@router.post("/{experiment_id}/run", status_code=202)
async def run_experiment(
    experiment_id: str,
    payload: Optional[ExperimentRunRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger an async experiment evaluation run via Celery.

    The task:
      1. Loads the experiment config and golden dataset
      2. Runs the RAG chain for each golden question (using config)
      3. Computes RAGAS evaluation scores
      4. Persists results and updates experiment status

    Returns immediately with 202 Accepted — poll /{id}/status for completion.
    """
    # Verify experiment exists
    exp = await exp_module.get_experiment(db, experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")

    if exp["status"] == "running":
        raise HTTPException(409, "Experiment is already running")

    # Prepare task kwargs
    config_override = payload.config_override if payload else None

    # Dispatch Celery task
    run_task = _get_run_task()
    task = run_task.delay(experiment_id, config_override=config_override)

    logger.info(
        f"Dispatched experiment {experiment_id} to Celery task {task.id}"
    )

    return {
        "experiment_id": experiment_id,
        "task_id": task.id,
        "message": "Experiment queued. Poll /api/experiments/{id}/status for results.",
    }


@router.get("/{experiment_id}/status", response_model=ExperimentStatusResponse)
async def get_experiment_status(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Poll the current status of an experiment.
    """
    exp = await exp_module.get_experiment(db, experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")

    return ExperimentStatusResponse(
        id=exp["id"],
        status=exp["status"],
        completed_at=exp.get("completed_at"),
    )
