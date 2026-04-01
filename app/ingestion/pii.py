"""PII (Personally Identifiable Information) detection and redaction.

Supports regex-based detection for:
- Email addresses
- Phone numbers (SG, US, international)
- NRIC/FIC (Singapore National ID)
- Credit card numbers
- IP addresses

Optionally uses Microsoft Presidio for advanced entity recognition.
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

_PATTERNS = {
    "EMAIL": {
        "pattern": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "description": "Email address",
    },
    "PHONE_SG": {
        "pattern": r"(?:\+65|65)?\s*(?:0|[1-9][0-9]{7})",
        "description": "Singapore phone number",
    },
    "PHONE_INT": {
        "pattern": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        "description": "International phone number",
    },
    "NRIC_SG": {
        "pattern": r"[STFG]\d{7}[A-Z]",
        "description": "Singapore NRIC/FIC",
    },
    "CREDIT_CARD": {
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "description": "Credit card number",
    },
    "IP_ADDRESS": {
        "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "description": "IP address",
    },
    "MAC_ADDRESS": {
        "pattern": r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}",
        "description": "MAC address",
    },
}

# ── Presidio integration ──────────────────────────────────────────────────────

PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    logger.warning("presidio-analyzer not available — using regex-only PII detection")


_analyzer: Optional[Any] = None
_anonymizer: Optional[Any] = None


def _get_presidio_analyzer() -> Any:
    global _analyzer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    return _analyzer


def _get_presidio_anonymizer() -> Any:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


# ── Public API ────────────────────────────────────────────────────────────────


def detect_pii(text: str, use_presidio: bool = False) -> List[Dict[str, Any]]:
    """
    Detect PII entities in text.

    Args:
        text:         Input text to scan.
        use_presidio: If True, also run Microsoft Presidio (slow but more accurate).

    Returns:
        List of dicts:
        [
            {
                "entity": str,   # e.g. "EMAIL", "NRIC_SG"
                "value": str,   # matched text
                "start": int,   # character offset (inclusive)
                "end": int,     # character offset (exclusive)
            },
            ...
        ]
    """
    if not text:
        return []

    entities = []

    # 1. Regex pass
    for entity_name, entity_info in _PATTERNS.items():
        for match in re.finditer(entity_info["pattern"], text):
            entities.append({
                "entity": entity_name,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

    # 2. Presidio pass (optional)
    if use_presidio and PRESIDIO_AVAILABLE:
        try:
            analyzer = _get_presidio_analyzer()
            presidio_results = analyzer.analyze(text=text, language="en")
            for r in presidio_results:
                # Deduplicate with regex findings
                entities.append({
                    "entity": r.entity_type,
                    "value": text[r.start:r.end],
                    "start": r.start,
                    "end": r.end,
                })
        except Exception as e:
            logger.warning(f"Presidio analysis failed: {e}")

    # Sort by position and remove exact duplicates
    entities.sort(key=lambda x: (x["start"], -x["end"]))
    seen = set()
    deduped = []
    for e in entities:
        key = (e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    logger.debug(f"Detected {len(deduped)} PII entities")
    return deduped


def redact_pii(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Redact PII entities in text by replacing with [REDACTED].

    Performs non-overlapping replacements from start to end.

    Args:
        text:     Input text.
        entities: List from detect_pii().

    Returns:
        Text with PII replaced by [REDACTED].
    """
    if not text or not entities:
        return text

    # Sort by start position
    sorted_entities = sorted(entities, key=lambda x: x["start"])

    result = []
    last_end = 0

    for entity in sorted_entities:
        start = entity["start"]
        end = entity["end"]

        # Skip overlapping entities
        if start < last_end:
            continue

        result.append(text[last_end:start])
        result.append("[REDACTED]")
        last_end = end

    result.append(text[last_end:])
    return "".join(result)


def detect_and_redact(text: str, use_presidio: bool = False) -> tuple[str, List[Dict[str, Any]]]:
    """
    Convenience function: detect PII then redact it.

    Returns:
        (redacted_text, entities_list)
    """
    entities = detect_pii(text, use_presidio=use_presidio)
    redacted = redact_pii(text, entities)
    return redacted, entities


def entities_to_json(entities: List[Dict[str, Any]]) -> str:
    """Serialize entities list to JSON string for DB storage."""
    return json.dumps(entities)
