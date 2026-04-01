"""
SafetyLayer — orchestrates guardrails, prompt injection detection,
grounding checks, and citation validation around the RAG chain.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.quality.citations import (
    CitationResult,
    validate_citations,
)
from app.quality.grounding import (
    GroundingResult,
    check_grounding,
)
from app.safety.guardrails import (
    GuardrailResult,
    validate_query,
)
from app.safety.prompt_injection import (
    InjectionResult,
    check_prompt_injection,
)

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    passed: bool
    stage: str  # "guardrails" | "injection" | "grounding" | "citation" | "chain"
    reason: Optional[str] = None
    details: Optional[Any] = None


@dataclass
class SafetyResponse:
    answer: str
    source_documents: List[Document]
    safety_results: List[SafetyResult] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_answer: Optional[str] = None

    @property
    def grounded(self) -> bool:
        for r in self.safety_results:
            if r.stage == "grounding":
                return r.details.grounded if r.details else True
        return True

    @property
    def citations_valid(self) -> bool:
        for r in self.safety_results:
            if r.stage == "citation":
                return r.details.valid if r.details else True
        return True


class SafetyLayer:
    """
    Wrapper that applies safety + quality checks to a RAG chain.

    Integrates:
      - Input guardrails
      - Prompt injection detection
      - Grounding verification
      - Citation validation

    Args:
        settings: Dict of safety flags drawn from RAGSettings or defaults.
        fallback_answer: What to return when safety checks fail.
    """

    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        fallback_answer: str = (
            "I'm sorry, but I couldn't generate a reliable answer "
            "based on the provided context. Please try rephrasing your question "
            "or providing more information."
        ),
    ):
        s = settings or {}
        self.guardrails_enabled = s.get("guardrails_enabled", True)
        self.injection_check_enabled = s.get("prompt_injection_check", True)
        self.grounding_check_enabled = s.get("grounding_check", True)
        self.citation_validation_enabled = s.get("citation_validation", True)
        self.confidence_threshold = s.get("confidence_threshold", 0.7)
        self.grounding_threshold = s.get("grounding_threshold", 0.7)
        self.injection_confidence_threshold = s.get("injection_confidence_threshold", 0.7)
        self.required_citation_threshold = s.get("required_citation_threshold", 0.5)
        self.max_retries = s.get("max_retries", 1)
        self.fallback_answer = fallback_answer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_query(self, query: str) -> Tuple[Optional[str], List[SafetyResult]]:
        """
        Run pre-chain safety checks on the user query.

        Returns (sanitized_query, list of SafetyResults).
        If guardrails fail, returns (None, [failed_result]).
        """
        results: List[SafetyResult] = []

        # 1. Guardrails
        gr_result = validate_query(
            query,
            guardrails_enabled=self.guardrails_enabled,
        )
        results.append(SafetyResult(
            passed=gr_result.passed,
            stage="guardrails",
            reason=None if gr_result.passed else gr_result.reason,
        ))
        if not gr_result.passed:
            return None, results

        # 2. Prompt injection
        inj_result = check_prompt_injection(
            query,
            injection_check_enabled=self.injection_check_enabled,
            confidence_threshold=self.injection_confidence_threshold,
        )
        results.append(SafetyResult(
            passed=not inj_result.detected,
            stage="injection",
            reason=inj_result.reason if inj_result.detected else None,
            details=inj_result,
        ))
        if inj_result.detected:
            logger.warning("Prompt injection detected: %s", inj_result.reason)
            # Don't block — log and warn, but don't reject

        sanitized = gr_result.sanitized_query or query
        return sanitized, results

    def check_response(
        self,
        question: str,
        answer: str,
        documents: List[Document],
        pre_check_results: Optional[List[SafetyResult]] = None,
    ) -> SafetyResponse:
        """
        Run post-chain quality checks on the generated answer.

        Args:
            question: Original user question.
            answer: LLM-generated answer string.
            documents: Retrieved + re-ranked source documents.
            pre_check_results: Results from check_query (to carry forward).

        Returns:
            SafetyResponse wrapping the answer with safety metadata.
        """
        results = list(pre_check_results) if pre_check_results else []
        fallback_triggered = False

        # 3. Grounding check
        grounding_result = check_grounding(
            question=question,
            answer=answer,
            documents=documents,
            grounding_check_enabled=self.grounding_check_enabled,
            threshold=self.grounding_threshold,
        )
        results.append(SafetyResult(
            passed=grounding_result.grounded,
            stage="grounding",
            reason=None if grounding_result.grounded else f"Grounding score: {grounding_result.overall_score:.2f}",
            details=grounding_result,
        ))

        if not grounding_result.grounded:
            logger.warning(
                "Grounding check failed (score=%.2f). Triggering fallback.",
                grounding_result.overall_score,
            )
            fallback_triggered = True

        # 4. Citation validation
        citation_result = validate_citations(
            answer=answer,
            documents=documents,
            citation_validation_enabled=self.citation_validation_enabled,
            required_citation_threshold=self.required_citation_threshold,
        )
        results.append(SafetyResult(
            passed=citation_result.valid,
            stage="citation",
            reason=None if citation_result.valid else citation_result.reason,
            details=citation_result,
        ))

        if not citation_result.valid:
            logger.warning(
                "Citation validation failed: %s",
                citation_result.reason,
            )

        return SafetyResponse(
            answer=answer,
            source_documents=documents,
            safety_results=results,
            fallback_triggered=fallback_triggered,
            fallback_answer=self.fallback_answer if fallback_triggered else None,
        )

    def invoke(
        self,
        question: str,
        chain_fn,  # callable that takes (question: str) → (answer: str, docs: List[Document])
        retries: int = 0,
    ) -> SafetyResponse:
        """
        Run the full safety-wrapped RAG flow.

        Args:
            question: User query.
            chain_fn: The actual RAG chain callable. Signature:
                (question: str) -> Tuple[str, List[Document]]
            retries: Current retry count (used internally).

        Returns:
            SafetyResponse with answer + safety metadata.
        """
        # Pre-chain checks
        sanitized, pre_results = self.check_query(question)
        if sanitized is None:
            return SafetyResponse(
                answer=self.fallback_answer,
                source_documents=[],
                safety_results=pre_results,
                fallback_triggered=True,
                fallback_answer=self.fallback_answer,
            )

        try:
            answer, documents = chain_fn(sanitized)
        except Exception as exc:
            logger.error("RAG chain failed: %s", exc)
            return SafetyResponse(
                answer=self.fallback_answer,
                source_documents=[],
                safety_results=pre_results,
                fallback_triggered=True,
                fallback_answer=self.fallback_answer,
            )

        # Post-chain checks
        response = self.check_response(question, answer, documents, pre_results)

        # Retry logic if grounding failed
        if response.fallback_triggered and retries < self.max_retries:
            logger.info("Retrying with relaxed threshold (attempt %d)", retries + 1)
            # Temporarily lower threshold for retry
            original_threshold = self.grounding_threshold
            self.grounding_threshold = max(0.0, self.grounding_threshold - 0.2)
            response = self.check_response(question, answer, documents, pre_results)
            self.grounding_threshold = original_threshold

        return response
