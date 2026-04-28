"""Generate a short conversation title using a lightweight LLM call."""
import logging

from app.services.llm_runner import llm_chat, LLMCircuitOpenError, LLMTimeoutError, LLMProviderError

logger = logging.getLogger(__name__)

TITLE_MODEL = "llama-3.1-8b-instant"
TITLE_PROVIDER = "groq"
TITLE_TEMPERATURE = 0.3
TITLE_TIMEOUT = 15.0


def _build_title_prompt(messages: list[dict]) -> str:
    """Build a single user prompt containing the conversation transcript."""
    lines = ["Given this conversation, suggest a short title (max 5 words). Only return the title, no explanation.", "", "Conversation:"]
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def generate_conversation_title(messages: list[dict]) -> str | None:
    """
    Generate a short title for a conversation based on its messages.

    Returns the generated title string, or None if generation failed.
    Failures are silent — callers should handle the None case by
    leaving the conversation title unchanged.
    """
    if not messages:
        return None

    prompt = _build_title_prompt(messages)
    try:
        result = await llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model=TITLE_MODEL,
            provider=TITLE_PROVIDER,
            temperature=TITLE_TEMPERATURE,
            timeout=TITLE_TIMEOUT,
        )
        title = result["choices"][0]["message"]["content"].strip()
        # Sanity-check: discard anything past the first line, strip quotes
        title = title.split("\n")[0].strip().strip('"').strip("'")
        if title:
            return title
        return None
    except (LLMCircuitOpenError, LLMTimeoutError, LLMProviderError) as e:
        logger.warning("Title generation failed: %s", e)
        return None
    except Exception as e:
        logger.warning("Title generation unexpected error: %s", e)
        return None
