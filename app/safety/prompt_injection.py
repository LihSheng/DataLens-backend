"""
Prompt injection detection — detects attempts to manipulate the LLM's behaviour
via adversarial instructions embedded in the user query.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Patterns associated with prompt injection attempts
# Not exhaustive — extend based on threat model
INJECTION_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    (
        "role_override",
        re.compile(r"(?i)(you are now|act as|pretend you are|roleplay|role-play)\s", re.IGNORECASE),
        "Role/identity override attempt",
    ),
    (
        "instruction_override",
        re.compile(
            r"(?i)(ignore (all )?previous|disregard (your )?instructions|"
            r"forget (your )?instructions|new instructions:?\s*\n)",
            re.IGNORECASE,
        ),
        "Instruction override attempt",
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"(?i)(show (me )?your|reveal|what are your|print your|summarize your)\s+"
            r"(system prompt|prompt|instructions|config|settings)",
            re.IGNORECASE,
        ),
        "System prompt leak attempt",
    ),
    (
        "prefix_hijack",
        re.compile(
            r"^(you are a|as an? |the following is|here is|in this conversation|, start)",
            re.IGNORECASE,
        ),
        "Prefix hijack (new conversation framing)",
    ),
    (
        "delimiter_injection",
        re.compile(r"^(system|user|assistant|human|bot)\s*:\s*\S", re.IGNORECASE),
        "Sohcket delimiter injection (role prefix)",
    ),
    (
        "hidden_text",
        re.compile(r"\x00|\x1b|\x08", re.IGNORECASE),  # null / escape / backspace
        "Hidden control characters",
    ),
    (
        "ssti",
        re.compile(r"\{\{|\}\}|\{\%|\%\}", re.IGNORECASE),
        "Template injection (SSTI) pattern",
    ),
    (
        "base64_inject",
        re.compile(r"(?i)[a-z0-9+/]{50,}={0,2}$"),
        "Long base64-like string (potential encoded payload)",
    ),
]


@dataclass
class InjectionResult:
    detected: bool
    category: Optional[str] = None
    reason: Optional[str] = None
    confidence: float = 0.0  # 0.0 – 1.0


def analyze(text: str) -> InjectionResult:
    """
    Scan a user query for prompt injection indicators.

    Returns InjectionResult with detected=True if any pattern fires.
    confidence is the fraction of patterns that matched (0.0 – 1.0).
    """
    hits: List[Tuple[str, str]] = []

    for category, pattern, reason in INJECTION_PATTERNS:
        if pattern.search(text):
            hits.append((category, reason))

    if not hits:
        return InjectionResult(detected=False, confidence=0.0)

    # Confidence = proportion of patterns that fired (capped at 1.0)
    confidence = min(len(hits) / max(len(INJECTION_PATTERNS) * 0.3, 1), 1.0)

    # If multiple patterns fire, escalate reason to generic "injection attempt"
    if len(hits) >= 2:
        return InjectionResult(
            detected=True,
            category="multi_pattern",
            reason=f"Multiple injection signals: {', '.join(r for _, r in hits)}",
            confidence=confidence,
        )

    category, reason = hits[0]
    return InjectionResult(
        detected=True,
        category=category,
        reason=reason,
        confidence=confidence,
    )


def check_prompt_injection(
    query: str,
    injection_check_enabled: bool = True,
    confidence_threshold: float = 0.7,
) -> InjectionResult:
    """
    Convenience wrapper — returns InjectionResult.

    Args:
        query: User input string.
        injection_check_enabled: Master kill-switch.
        confidence_threshold: Minimum confidence to flag (0.0 – 1.0).
    """
    if not injection_check_enabled:
        return InjectionResult(detected=False, confidence=0.0)

    result = analyze(query)
    if result.confidence < confidence_threshold:
        return InjectionResult(detected=False, confidence=result.confidence)

    return result
