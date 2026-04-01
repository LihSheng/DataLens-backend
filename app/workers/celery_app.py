"""
Celery application — Stage 2 (updated in Stage 6).

Full implementation: async Celery tasks for ingestion (Stage 2) and
experiment evaluation (Stage 6).
"""
import json
import logging
from typing import Any, Dict, List, Optional

from celery import Celery

from app.config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.ingestion_worker",
        "app.workers.evaluation_worker",  # Stage 6 experiment tasks
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


# ─────────────────────────────────────────────────────────
# Stage 6: Experiment evaluation task
# ─────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="experiments.run")
def run_experiment_task(
    self,
    experiment_id: str,
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Async Celery task that runs a full experiment evaluation.

    Steps:
      1. Load experiment config and golden dataset.
      2. For each golden entry, invoke the RAG chain with the experiment config.
      3. Run RAGAS evaluation on all (question, answer, contexts) pairs.
      4. Persists ExperimentResult rows and aggregated scores.

    Args:
        experiment_id: The experiment UUID.
        config_override: Optional per-run config overrides merged into the
                         experiment's stored config.

    Returns:
        Dict with experiment_id, status, and aggregated results.
    """
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.db.session import AsyncSessionLocal
    from app.evaluation import experiment as exp_module
    from app.evaluation.golden_dataset import get_all_golden_entries
    from app.chains.rag_chain import RAGChain

    logger = logging.getLogger(__name__)
    logger.info(f"[Experiment {experiment_id}] Starting evaluation task")

    async def _run() -> Dict[str, Any]:
        async with AsyncSessionLocal() as db:
            # Load experiment
            exp = await exp_module.get_experiment(db, experiment_id)
            if exp is None:
                raise ValueError(f"Experiment {experiment_id} not found")

            config = dict(exp["config"])
            if config_override:
                config.update(config_override)

            metrics = config.get("metrics")

            # Load golden dataset
            golden_rows = await get_all_golden_entries(db)
            golden_entries = [r.to_dict() for r in golden_rows]

            if not golden_entries:
                logger.warning(f"Experiment {experiment_id}: golden dataset is empty")
                await exp_module.update_experiment_status(
                    db, experiment_id, "completed",
                    results_json=json.dumps({"count": 0, "error": "empty_golden_dataset"})
                )
                return {"experiment_id": experiment_id, "status": "completed", "results": {"count": 0}}

            # Generate answers for each golden question
            generated_answers: List[Dict[str, Any]] = []
            for golden in golden_entries:
                try:
                    question = golden["question"]

                    # Build RAGChain from experiment config
                    chain = RAGChain(
                        settings={
                            "query_expansion": config.get("query_expansion", False),
                            "hyde": config.get("hyde", False),
                            "reranker": config.get("reranker", False),
                            "confidence_threshold": config.get("confidence_threshold", 0.7),
                            # Memory disabled for eval
                            "enable_memory": False,
                            "enable_followup": False,
                        },
                        filters={
                            k: v for k, v in config.items()
                            if k in ("document_ids", "sources", "tags") and v
                        } or None,
                        k=config.get("k", 8),
                        rerank_top_n=config.get("rerank_top_n", 4),
                    )

                    result = chain.invoke({"question": question})
                    retrieved_contexts = [
                        doc.page_content for doc in result.get("source_documents", [])
                    ]

                    generated_answers.append({
                        "golden_id": golden["id"],
                        "generated_answer": result["answer"],
                        "retrieved_contexts": retrieved_contexts,
                    })

                except Exception as e:
                    logger.error(
                        f"[Experiment {experiment_id}] Failed on question "
                        f"'{golden['question'][:50]}...': {e}"
                    )
                    generated_answers.append({
                        "golden_id": golden["id"],
                        "generated_answer": "",
                        "retrieved_contexts": [],
                    })

            # Run RAGAS evaluation
            aggregate = await exp_module.run_experiment(
                db=db,
                experiment_id=experiment_id,
                golden_entries=golden_entries,
                generated_answers=generated_answers,
                metrics=metrics,
            )

            return {
                "experiment_id": experiment_id,
                "status": "completed",
                "results": aggregate,
            }

    try:
        result = asyncio.run(_run())
        logger.info(f"[Experiment {experiment_id}] Completed successfully")
        return result
    except Exception as e:
        logger.error(f"[Experiment {experiment_id}] Failed: {e}", exc_info=True)

        async def _fail():
            async with AsyncSessionLocal() as db:
                await exp_module.update_experiment_status(
                    db, experiment_id, "failed"
                )

        asyncio.run(_fail())
        raise
