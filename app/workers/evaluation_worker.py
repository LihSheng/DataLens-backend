"""
Evaluation worker — Stage 6.

Entry point for Celery worker processes that handle experiment evaluation tasks.
Importing this module registers the `run_experiment_task` with the Celery app.

Run with:
    celery -A app.workers.evaluation_worker worker --loglevel=info
"""
from app.workers.celery_app import celery_app, run_experiment_task

__all__ = ["celery_app", "run_experiment_task"]
