"""
Grounding checker — verifies that LLM answers are factually supported by the
retrieved context documents.

Uses an LLM (or heuristic) to score how well each claim in the answer
is grounded in the provided documents.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.services.vectorstore_service import get_llm

logger = logging.getLogger(__name__)

DEFAULT_GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an answer quality evaluator. Given a question, an answer, "
        "and a list of source documents, determine how well each claim in the answer "
        "is supported by the documents.\n"
        "Rate each claim on a score of 0.0 (completely unsupported) to 1.0 (fully supported).\n"
        "If the answer says 'I don't know' or 'not enough context' and the documents "
        "also don't contain the information, that is valid — score it 1.0.\n"
        "Respond ONLY with a valid JSON object: "
        '{"claims": [{"text": "...", "supported": true/false, "score": 0.0-1.0, "reason": "..."}], '
        '"overall_score": 0.0-1.0, "verdict": "grounded|partial|ungrounded"}',
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Documents:\n{documents}\n\n"
        "Evaluate grounding now — JSON only.",
    ),
])


@dataclass
class ClaimCheck:
    text: str
    supported: bool
    score: float
    reason: str


@dataclass
class GroundingResult:
    grounded: bool
    overall_score: float  # 0.0 – 1.0
    verdict: str  # "grounded" | "partial" | "ungrounded"
    claims: List[ClaimCheck]
    reason: Optional[str] = None


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(No documents provided)"
    return "\n\n---\n\n".join(
        f"[Source {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def _split_into_claims(text: str) -> List[str]:
    """
    Simple sentence-level split. For production, consider dependency parsing.
    """
    import re
    # Split on sentence boundaries, newlines, or bullet markers
    sentences = re.split(r"(?<=[.!?])\s+|\n\s*[-•*]\s*|\n+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


class GroundingChecker:
    """
    Evaluate whether an LLM answer is grounded in the retrieved context.

    Args:
        threshold: Minimum overall_score to consider the answer "grounded".
                   Below this → fallback triggered.
        use_llm: Use LLM for evaluation. If False, uses keyword overlap heuristic.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_llm: bool = True,
    ):
        self.threshold = threshold
        self.use_llm = use_llm

    def check(
        self,
        question: str,
        answer: str,
        documents: List[Document],
    ) -> GroundingResult:
        """
        Evaluate grounding of answer against documents.

        Args:
            question: The original user question.
            answer: The LLM-generated answer.
            documents: Retrieved source documents.

        Returns:
            GroundingResult with verdict and per-claim breakdown.
        """
        if not answer or not answer.strip():
            return GroundingResult(
                grounded=False,
                overall_score=0.0,
                verdict="ungrounded",
                claims=[],
                reason="Answer is empty.",
            )

        if not documents:
            # No docs → can't ground → treat "I don't know" as valid
            if _looks_like_unknown_answer(answer):
                return GroundingResult(
                    grounded=True,
                    overall_score=1.0,
                    verdict="grounded",
                    claims=[],
                    reason="No documents; answer correctly indicates uncertainty.",
                )
            return GroundingResult(
                grounded=False,
                overall_score=0.0,
                verdict="ungrounded",
                claims=[],
                reason="No source documents provided.",
            )

        if not self.use_llm:
            return self._heuristic_check(question, answer, documents)

        return self._llm_check(question, answer, documents)

    def _llm_check(
        self,
        question: str,
        answer: str,
        documents: List[Document],
    ) -> GroundingResult:
        try:
            llm = get_llm()
            chain = DEFAULT_GROUNDING_PROMPT | llm | JsonOutputParser()

            docs_text = _format_docs(documents)
            raw = chain.invoke({
                "question": question,
                "answer": answer,
                "documents": docs_text,
            })

            claims = [
                ClaimCheck(
                    text=c.get("text", ""),
                    supported=c.get("supported", False),
                    score=c.get("score", 0.0),
                    reason=c.get("reason", ""),
                )
                for c in raw.get("claims", [])
            ]

            overall_score = raw.get("overall_score", 0.0)
            verdict = raw.get("verdict", "ungrounded")
            grounded = overall_score >= self.threshold

            return GroundingResult(
                grounded=grounded,
                overall_score=overall_score,
                verdict=verdict,
                claims=claims,
            )

        except Exception as exc:
            logger.warning("Grounding check LLM failed, falling back to heuristic: %s", exc)
            return self._heuristic_check(question, answer, documents)

    def _heuristic_check(
        self,
        question: str,
        answer: str,
        documents: List[Document],
    ) -> GroundingResult:
        """
        Keyword-overlap fallback when LLM is unavailable.
        Counts what fraction of answer words appear in the source docs.
        """
        import re

        answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "i", "you", "he", "she", "it", "we", "they",
            "their", "this", "that", "these", "those", "and", "or", "but", "if",
            "not", "no", "so", "what", "which", "who", "how", "when", "where",
            "why", "according", "based", "context", "question", "answer",
        }
        answer_words -= stopwords

        if not answer_words:
            return GroundingResult(
                grounded=True,
                overall_score=1.0,
                verdict="grounded",
                claims=[],
                reason="No substantive words to check (heuristic).",
            )

        all_doc_text = " ".join(doc.page_content.lower() for doc in documents)
        matched = sum(1 for w in answer_words if w in all_doc_text)
        score = matched / len(answer_words)

        grounded = score >= self.threshold
        verdict = "grounded" if score >= 0.8 else "partial" if score >= 0.4 else "ungrounded"

        return GroundingResult(
            grounded=grounded,
            overall_score=score,
            verdict=verdict,
            claims=[],
            reason=f"Heuristic overlap score: {score:.2f}",
        )


def check_grounding(
    question: str,
    answer: str,
    documents: List[Document],
    grounding_check_enabled: bool = True,
    threshold: float = 0.7,
) -> GroundingResult:
    """
    Convenience function.
    """
    if not grounding_check_enabled:
        return GroundingResult(grounded=True, overall_score=1.0, verdict="grounded", claims=[])

    checker = GroundingChecker(threshold=threshold)
    return checker.check(question, answer, documents)


def _looks_like_unknown_answer(text: str) -> bool:
    import re
    patterns = [
        r"i don't know",
        r"i do not know",
        r"not enough information",
        r"cannot (be )?determined",
        r"no (sufficient |relevant )?information",
        r"not (provided|available|given)",
        r"the context does(n't| not) contain",
    ]
    return any(re.search(p, text.lower()) for p in patterns)
