"""Evaluation module — Stage 6."""
from app.evaluation.ragas_eval import (
    evaluate_answer,
    batch_evaluate,
    EvaluationResult,
)
from app.evaluation import golden_dataset
from app.evaluation import experiment

__all__ = [
    "evaluate_answer",
    "batch_evaluate",
    "EvaluationResult",
    "golden_dataset",
    "experiment",
]
