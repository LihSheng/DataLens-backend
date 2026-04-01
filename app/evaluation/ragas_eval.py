"""
RAGAS evaluation module — Stage 6.

Provides async evaluation of RAG answers using the RAGAS library.
Supports: faithfulness, answer_correctness, answer_relevance, context_precision,
context_recall, context_relevancy.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# RAGAS imports — require `ragas` package
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_correctness,
        answer_relevance,
        context_precision,
        context_recall,
        context_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("ragas not installed — evaluation will be a no-op stub")


@dataclass
class EvaluationResult:
    """Container for a single question's evaluation scores."""
    question: str
    expected_answer: Optional[str]
    generated_answer: str
    retrieved_contexts: List[str]
    faithfulness: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_relevancy: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_metrics_json(self) -> str:
        """Return just the metrics portion as a JSON string."""
        return json.dumps({
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "context_relevancy": self.context_relevancy,
        })


async def evaluate_answer(
    question: str,
    generated_answer: str,
    retrieved_contexts: List[str],
    expected_answer: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Evaluate a single RAG answer using RAGAS.

    Args:
        question: The input question.
        generated_answer: The RAG system's generated answer.
        retrieved_contexts: List of retrieved context strings.
        expected_answer: Ground-truth answer (optional, needed for answer_correctness).
        metrics: List of metric names to compute. Defaults to all available.

    Returns:
        EvaluationResult with scores.
    """
    if not RAGAS_AVAILABLE:
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            error="ragas_not_available",
        )

    # Default metrics
    if metrics is None:
        metrics = ["faithfulness", "answer_relevance", "context_relevancy"]
        if expected_answer:
            metrics.extend(["answer_correctness", "context_recall"])

    # Build dataset
    data = {
        "user_input": [question],
        "response": [generated_answer],
        "retrieved_contexts": [retrieved_contexts],
    }
    if expected_answer:
        data["reference"] = [expected_answer]

    try:
        dataset = Dataset.from_dict(data)

        # Select metric objects
        metric_objs = []
        for m in metrics:
            if m == "faithfulness":
                metric_objs.append(faithfulness)
            elif m == "answer_correctness":
                metric_objs.append(answer_correctness)
            elif m == "answer_relevance":
                metric_objs.append(answer_relevance)
            elif m == "context_precision":
                metric_objs.append(context_precision)
            elif m == "context_recall":
                metric_objs.append(context_recall)
            elif m == "context_relevancy":
                metric_objs.append(context_relevancy)

        result = evaluate(dataset, metrics=metric_objs)
        scores = result.to_pandas().iloc[0].to_dict()

        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            faithfulness=_safe_float(scores.get("faithfulness")),
            answer_correctness=_safe_float(scores.get("answer_correctness")),
            answer_relevance=_safe_float(scores.get("answer_relevance")),
            context_precision=_safe_float(scores.get("context_precision")),
            context_recall=_safe_float(scores.get("context_recall")),
            context_relevancy=_safe_float(scores.get("context_relevancy")),
        )
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            retrieved_contexts=retrieved_contexts,
            error=str(e),
        )


async def batch_evaluate(
    items: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> List[EvaluationResult]:
    """
    Evaluate a batch of (question, answer, contexts, expected_answer) items.

    Args:
        items: List of dicts with keys: question, generated_answer,
               retrieved_contexts (List[str]), expected_answer (optional).
        metrics: Metrics to compute.

    Returns:
        List of EvaluationResult.
    """
    results = []
    for item in items:
        result = await evaluate_answer(
            question=item["question"],
            generated_answer=item["generated_answer"],
            retrieved_contexts=item.get("retrieved_contexts", []),
            expected_answer=item.get("expected_answer"),
            metrics=metrics,
        )
        results.append(result)
    return results


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float or return None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
