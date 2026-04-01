"""
Citation validator — checks whether citations/references in the LLM answer
actually correspond to valid source documents that were retrieved.

Detects:
  - Citation markers that reference non-existent sources
  - Claims in answer not backed by any cited document
  - Missing citations for factual statements
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Match bracketed citation patterns: [Source: X], [1], [Source 1], (Source: X)
CITATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\[Source:\s*([^\]]+)\]"),
    re.compile(r"(?i)\[Source\s+(\d+)\]"),
    re.compile(r"(?i)\[(\d+)\]"),                          # bare [1]
    re.compile(r"(?i)\(Source:\s*([^)]+)\)"),
    re.compile(r"(?i)\(see\s+source\s+(\d+)\)", re.IGNORECASE),
]


@dataclass
class CitationCheck:
    marker: str          # e.g. "[Source: foo]"
    referenced_source: str
    found_in_docs: bool
    content_aligned: bool
    reason: str


@dataclass
class CitationResult:
    valid: bool
    checked: List[CitationCheck]
    missing_citations: List[str]  # factual claims without any citation
    overall_score: float  # 0.0 – 1.0
    verdict: str  # "valid" | "partial" | "invalid"
    reason: Optional[str] = None


def _extract_citations(text: str) -> List[Tuple[str, str]]:
    """
    Extract all citation markers and the source they reference.
    Returns list of (marker, source_id).
    """
    results: List[Tuple[str, str]] = []
    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            marker = match.group(0)
            # Group 1 always holds the source identifier
            source = match.group(1).strip()
            results.append((marker, source))
    return results


def _source_matches(source_id: str, doc: Document) -> bool:
    """Check if a citation source ID matches a document's metadata."""
    meta = doc.metadata or {}
    # Try various metadata fields
    for field in ("source", "id", "name", "title", "filename"):
        if meta.get(field) and source_id.lower() in str(meta[field]).lower():
            return True
    # Also check page_content for source mentions
    if source_id.lower() in doc.page_content.lower():
        return True
    return False


def _claim_has_citation(claim: str, citations: List[Tuple[str, str]]) -> bool:
    """Rough check: does this claim mention any cited source?"""
    claim_lower = claim.lower()
    for _, source in citations:
        if source.lower() in claim_lower:
            return True
    return False


def _split_into_claims(text: str) -> List[str]:
    """Split answer into individual claims (sentence-level)."""
    return re.split(r"(?<=[.!?])\s+", text)


class CitationValidator:
    """
    Validate that citations in the LLM answer are backed by real source documents.

    Args:
        required_citation_threshold: Fraction of factual sentences that should
            carry a citation (0.0 – 1.0). Below this → "partial" verdict.
    """

    def __init__(
        self,
        required_citation_threshold: float = 0.5,
    ):
        self.required_citation_threshold = required_citation_threshold

    def validate(
        self,
        answer: str,
        documents: List[Document],
        question: Optional[str] = None,
    ) -> CitationResult:
        """
        Check citation validity in the answer.

        Args:
            answer: LLM-generated answer (may contain citation markers).
            documents: Retrieved source documents.
            question: Optional question (for missing-citation heuristic).

        Returns:
            CitationResult with per-citation checks and overall verdict.
        """
        if not documents:
            return CitationResult(
                valid=False,
                checked=[],
                missing_citations=[],
                overall_score=0.0,
                verdict="invalid",
                reason="No source documents provided.",
            )

        citations = _extract_citations(answer)
        checks: List[CitationCheck] = []

        for marker, source_id in citations:
            # Find matching doc
            matched_doc = next(
                (d for d in documents if _source_matches(source_id, d)), None
            )
            found = matched_doc is not None

            # Content alignment: rough keyword overlap
            aligned = False
            reason = "not found in any source"
            if matched_doc:
                aligned = source_id.lower() in matched_doc.page_content.lower()
                reason = "found and content-aligned" if aligned else "found but content mismatch"

            checks.append(CitationCheck(
                marker=marker,
                referenced_source=source_id,
                found_in_docs=found,
                content_aligned=aligned,
                reason=reason,
            ))

        # Compute score
        if not citations:
            # No citations at all
            score = 0.0
            verdict = "invalid"
            reason = "No citations found in answer."
        else:
            valid_fraction = sum(1 for c in checks if c.found_in_docs and c.content_aligned) / len(checks)
            score = valid_fraction
            verdict = "valid" if valid_fraction >= 0.9 else "partial" if valid_fraction >= 0.5 else "invalid"
            reason = f"{sum(1 for c in checks if c.found_in_docs)}/{len(checks)} citations resolved."

        # Check for missing citations on factual sentences
        missing: List[str] = []
        if citations:
            sentences = _split_into_claims(answer)
            # A sentence is "factual" if it has > 5 words and isn't a question/statement about not knowing
            factual = [s for s in sentences if len(s.split()) > 5 and not _is_non_factual(s)]
            cited = [s for s in factual if _claim_has_citation(s, citations)]
            if factual and (len(cited) / len(factual)) < self.required_citation_threshold:
                # Only report if there are long factual sentences lacking citations
                lacking = [s for s in factual if not _claim_has_citation(s, citations)]
                missing = lacking[:3]  # cap at 3 examples

        return CitationResult(
            valid=verdict == "valid",
            checked=checks,
            missing_citations=missing,
            overall_score=score,
            verdict=verdict,
            reason=reason,
        )


def validate_citations(
    answer: str,
    documents: List[Document],
    citation_validation_enabled: bool = True,
    required_citation_threshold: float = 0.5,
) -> CitationResult:
    """
    Convenience function.
    """
    if not citation_validation_enabled:
        return CitationResult(valid=True, checked=[], missing_citations=[], overall_score=1.0, verdict="valid")

    validator = CitationValidator(required_citation_threshold=required_citation_threshold)
    return validator.validate(answer, documents)


def _is_non_factual(sentence: str) -> bool:
    s = sentence.lower()
    non_factual_phrases = [
        "i don't know", "i do not know", "i'm not sure", "i cannot determine",
        "no information", "not enough", "the context does not",
        "as an ai", "i'm an ai", "i am an ai",
    ]
    return any(phrase in s for phrase in non_factual_phrases)
