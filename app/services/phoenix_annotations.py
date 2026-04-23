"""
Phoenix Annotations Service — Stage 3.

Handles two concerns:
1. Live RAGAS evaluation — async scoring after each RAG answer, POSTed to
   Phoenix as span annotations (Faithfulness, Answer Relevance, Context Precision).
2. Human feedback — POSTed to Phoenix as a "Human Feedback" span annotation.

Both are fire-and-forget (non-blocking) so they don't slow down the chat response.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ─── HTTP client ───────────────────────────────────────────────────────────────

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.phoenix_base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    return _client


# ─── Core annotation helpers ──────────────────────────────────────────────────

async def _post_annotation(
    trace_id: str,
    span_id: str | None,
    name: str,
    label: Any,
    metadata: Dict[str, Any] | None = None,
) -> bool:
    """
    POST a single span annotation to Phoenix /v1/span_annotations.

    Returns True on success, False on failure (never raises).
    """
    payload: Dict[str, Any] = {
        "trace_id": trace_id,
        "name": name,
        "label": label,
    }
    if span_id:
        payload["span_id"] = span_id
    if metadata:
        payload["metadata"] = metadata

    if not settings.phoenix_enabled:
        return False

    try:
        client = await _get_client()
        response = await client.post("/v1/span_annotations", json=payload)
        response.raise_for_status()
        logger.debug(f"Phoenix annotation posted: {name} = {label} for trace {trace_id}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to post Phoenix annotation {name}: {exc}")
        return False


# ─── RAGAS live evaluation ────────────────────────────────────────────────────

def _extract_contexts(source_docs: List[Any]) -> List[str]:
    """Normalise source documents to a list of context strings."""
    if not source_docs:
        return []
    if isinstance(source_docs[0], str):
        return source_docs
    return [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in source_docs
    ]


async def run_live_ragas_eval(
    question: str,
    answer: str,
    source_docs: List[Any],
    trace_id: str,
    span_id: str | None = None,
) -> None:
    """
    Compute RAGAS metrics for a live RAG answer and POST annotations to Phoenix.

    Metrics computed:
      - faithfulness       (always)
      - answer_relevance   (always)
      - context_precision  (always)

    This function is async but is meant to be fire-and-forget (no await needed
    on the call site).  Failures are logged and never propagate.
    """
    # Import RAGAS lazily so the rest of the app works even if ragas isn't installed
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevance, context_precision
        from datasets import Dataset
    except ImportError:
        logger.debug("ragas not installed — skipping live RAGAS eval")
        return

    contexts = _extract_contexts(source_docs)
    if not contexts:
        logger.debug("No contexts — skipping live RAGAS eval")
        return

    data = {
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
    }

    try:
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance, context_precision],
        )
        scores = result.to_pandas().iloc[0]

        metrics_to_post = [
            ("Faithfulness", scores.get("faithfulness")),
            ("Answer Relevance", scores.get("answer_relevance")),
            ("Context Precision", scores.get("context_precision")),
        ]

        for name, value in metrics_to_post:
            if value is not None:
                try:
                    label = round(float(value), 4)
                except (TypeError, ValueError):
                    label = str(value)
                await _post_annotation(
                    trace_id=trace_id,
                    span_id=span_id,
                    name=name,
                    label=label,
                    metadata={"score": label},
                )

    except Exception as exc:
        logger.warning(f"Live RAGAS eval failed for trace {trace_id}: {exc}")


# ─── Human feedback ───────────────────────────────────────────────────────────

async def submit_feedback(
    trace_id: str,
    span_id: str | None,
    label: str,  # "positive" | "negative"
) -> bool:
    """
    Submit a human feedback annotation to Phoenix.

    label must be "positive" or "negative".
    Returns True on success, False on failure.
    """
    if label not in ("positive", "negative"):
        raise ValueError("label must be 'positive' or 'negative'")

    return await _post_annotation(
        trace_id=trace_id,
        span_id=span_id,
        name="Human Feedback",
        label=label,
        metadata={"source": "datalens-ui"},
    )
