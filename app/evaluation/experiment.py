"""
Experiment framework — Stage 6.

Manages experiment lifecycle: create, configure, run evaluation,
compare results against golden dataset.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.experiment import Experiment, ExperimentResult
from app.evaluation.ragas_eval import batch_evaluate, EvaluationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Experiment CRUD
# ─────────────────────────────────────────────────────────

async def create_experiment(
    db: AsyncSession,
    name: str,
    config: Dict[str, Any],
    description: Optional[str] = None,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new experiment record.

    Args:
        db: Async SQLAlchemy session.
        name: Human-readable experiment name.
        config: JSON-serializable dict with retrieval/chain settings to test.
        description: Optional description.
        created_by: User ID who created it.

    Returns:
        The created experiment as a dict.
    """
    experiment = Experiment(
        name=name,
        description=description,
        config_json=json.dumps(config),
        status="pending",
        created_by=created_by,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return _experiment_to_dict(experiment)


async def list_experiments(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[Dict[str, Any]], int]:
    """
    List all experiments, newest first.

    Returns:
        (list of experiment dicts, total count).
    """
    from sqlalchemy import func
    stmt = (
        select(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    results = await db.execute(stmt)
    experiments = results.scalars().all()

    count_result = await db.execute(select(func.count(Experiment.id)))
    total = count_result.scalar() or 0

    return [_experiment_to_dict(e) for e in experiments], total


async def get_experiment(
    db: AsyncSession,
    experiment_id: str,
) -> Optional[Dict[str, Any]]:
    """Get a single experiment with its results."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        return None

    # Load results
    results_result = await db.execute(
        select(ExperimentResult)
        .where(ExperimentResult.experiment_id == experiment_id)
        .order_by(ExperimentResult.created_at)
    )
    results_rows = results_result.scalars().all()

    exp_dict = _experiment_to_dict(experiment)
    exp_dict["results"] = [_result_to_dict(r) for r in results_rows]
    return exp_dict


async def update_experiment_status(
    db: AsyncSession,
    experiment_id: str,
    status: str,
    results_json: Optional[str] = None,
) -> None:
    """Update experiment status and optionally aggregate results."""
    values: Dict[str, Any] = {"status": status}
    if status in ("completed", "failed"):
        values["completed_at"] = datetime.utcnow()
    if results_json is not None:
        values["results_json"] = results_json

    await db.execute(
        update(Experiment)
        .where(Experiment.id == experiment_id)
        .values(**values)
    )
    await db.commit()


# ─────────────────────────────────────────────────────────
# Run experiment
# ─────────────────────────────────────────────────────────

async def run_experiment(
    db: AsyncSession,
    experiment_id: str,
    golden_entries: List[Dict[str, Any]],
    generated_answers: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run evaluation for an experiment.

    This orchestrates:
      1. Generating answers (caller provides these per golden entry)
      2. Running RAGAS evaluation
      3. Persisting ExperimentResult rows
      4. Aggregating final scores into Experiment.results_json

    Args:
        db: Async SQLAlchemy session.
        experiment_id: The experiment to run.
        golden_entries: List of {id, question, answer, context}.
        generated_answers: List of {golden_id, generated_answer, retrieved_contexts}.
        metrics: RAGAS metrics to compute.

    Returns:
        Aggregated results dict.
    """
    # Update status to running
    await update_experiment_status(db, experiment_id, "running")

    try:
        # Build evaluation items — match generated answers to golden entries
        eval_items: List[Dict[str, Any]] = []
        answer_map = {a["golden_id"]: a for a in generated_answers}

        for entry in golden_entries:
            golden_id = entry["id"]
            answer_data = answer_map.get(golden_id, {})
            eval_items.append({
                "question": entry["question"],
                "expected_answer": entry["answer"],
                "generated_answer": answer_data.get("generated_answer", ""),
                "retrieved_contexts": answer_data.get("retrieved_contexts", []),
            })

        # Run RAGAS evaluation
        eval_results: List[EvaluationResult] = await batch_evaluate(
            items=eval_items,
            metrics=metrics,
        )

        # Persist individual results
        for item, eval_result in zip(eval_items, eval_results):
            result_row = ExperimentResult(
                experiment_id=experiment_id,
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                generated_answer=eval_result.generated_answer,
                retrieved_contexts=eval_result.retrieved_contexts,
                metrics_json=eval_result.to_metrics_json(),
            )
            db.add(result_row)

        # Aggregate scores
        aggregate = _aggregate_results(eval_results)
        results_json = json.dumps(aggregate)

        await db.commit()

        # Update experiment with aggregated results and completed status
        await update_experiment_status(
            db, experiment_id, "completed", results_json=results_json
        )

        return aggregate

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
        await update_experiment_status(db, experiment_id, "failed")
        raise


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _experiment_to_dict(exp: Experiment) -> Dict[str, Any]:
    return {
        "id": exp.id,
        "name": exp.name,
        "description": exp.description,
        "config": json.loads(exp.config_json) if exp.config_json else {},
        "results": json.loads(exp.results_json) if exp.results_json else None,
        "status": exp.status,
        "created_by": exp.created_by,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "completed_at": (
            exp.completed_at.isoformat() if exp.completed_at else None
        ),
    }


def _result_to_dict(r: ExperimentResult) -> Dict[str, Any]:
    return {
        "id": r.id,
        "experiment_id": r.experiment_id,
        "question": r.question,
        "expected_answer": r.expected_answer,
        "generated_answer": r.generated_answer,
        "retrieved_contexts": r.retrieved_contexts,
        "metrics": json.loads(r.metrics_json) if r.metrics_json else {},
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Compute mean scores across all evaluated items."""
    count = len(results)
    if count == 0:
        return {"count": 0}

    def mean(key: str) -> Optional[float]:
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "count": count,
        "faithfulness": mean("faithfulness"),
        "answer_correctness": mean("answer_correctness"),
        "answer_relevance": mean("answer_relevance"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "context_relevancy": mean("context_relevancy"),
    }
