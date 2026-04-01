"""
Input guardrails — validate and sanitize user queries before they reach the LLM.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Set

# --- Configurable limits ---
MAX_QUERY_LENGTH = 2000
MAX_DOC_RETURN = 20
BLOCKED_PATTERNS: Set[re.Pattern] = {
    # Generic SQL injection heuristics
    re.compile(r"('\s*OR\s*'1'\s*=\s*'1)", re.IGNORECASE),
    re.compile(r"('\s*OR\s*1\s*=\s*1)", re.IGNORECASE),
    re.compile(r";\s*DROP\s+TABLE", re.IGNORECASE),
    re.compile(r";\s*DELETE\s+FROM", re.IGNORECASE),
    re.compile(r";\s*INSERT\s+INTO", re.IGNORECASE),
    re.compile(r";\s*UPDATE\s+\w+\s+SET", re.IGNORECASE),
    # Shell / code injection
    re.compile(r"(\|\s*nc\s)", re.IGNORECASE),
    re.compile(r"(`[^`]+`)", re.IGNORECASE),
    re.compile(r"\$\([^)]+\)", re.IGNORECASE),
    re.compile(r"\{\{[^}]+\}\}", re.IGNORECASE),  # SSTI
}
# Simple word blocklist
BLOCKED_WORDS: Set[str] = {
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your instructions",
    "you are now in",
    "pretend you are",
    "reveal your system prompt",
    "show your prompt",
    "replace your instructions",
}


@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    sanitized_query: Optional[str] = None


def check_blocked_patterns(query: str) -> Optional[str]:
    """Return reason string if any blocked pattern fires, else None."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(query):
            return f"Blocked pattern detected: {pattern.pattern!r}"
    q_lower = query.lower()
    for word in BLOCKED_WORDS:
        if word in q_lower:
            return f"Blocked word detected: {word!r}"
    return None


def sanitize_query(query: str) -> str:
    """Strip potentially dangerous characters, trim length."""
    # Remove null bytes
    query = query.replace("\x00", "")
    # Collapse excessive whitespace
    query = re.sub(r"\s+", " ", query)
    # Remove control characters
    query = "".join(ch for ch in query if ch.isprintable() or ch in "\t\n")
    return query.strip()


def validate_query(
    query: str,
    max_length: int = MAX_QUERY_LENGTH,
    guardrails_enabled: bool = True,
) -> GuardrailResult:
    """
    Run all guardrail checks on a user query.

    Args:
        query: Raw user input.
        max_length: Maximum allowed query length in characters.
        guardrails_enabled: Master kill-switch from settings.

    Returns:
        GuardrailResult with passed=True if all checks pass.
    """
    if not guardrails_enabled:
        return GuardrailResult(passed=True)

    if not query or not query.strip():
        return GuardrailResult(passed=False, reason="Query is empty.")

    if len(query) > max_length:
        return GuardrailResult(
            passed=False,
            reason=f"Query exceeds maximum length ({len(query)} > {max_length}).",
        )

    # Pattern + word checks
    reason = check_blocked_patterns(query)
    if reason:
        return GuardrailResult(passed=False, reason=reason)

    sanitized = sanitize_query(query)
    return GuardrailResult(passed=True, sanitized_query=sanitized)
