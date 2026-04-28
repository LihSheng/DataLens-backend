"""
LLM invocation wrapper with timeout + circuit breaker.

Usage:
    result = await llm_chat(messages, model="llama-3.3-70b-versatile", provider="groq", timeout=60)

The circuit breaker trips after 5 consecutive failures and stays open for 30s,
preventing cascade failures when the LLM provider is down.

For LangChain-based chains (rag_chain.py), use the circuit breaker by calling
check_circuit_breaker(provider) before invoking the chain, or use the
timeout-enabled LLM from get_llm().
"""
import time
from typing import Optional

# ─── Circuit breaker state ──────────────────────────────────────────────────

class CircuitBreaker:
    """
    Simple circuit breaker for LLM provider calls.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """

    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 30  # seconds before trying again

    def __init__(self):
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed | open | half_open

    def should_try(self) -> bool:
        now = time.time()
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._last_failure_time and (now - self._last_failure_time) >= self.RECOVERY_TIMEOUT:
                self._state = "half_open"
                return True
            return False
        # half_open — always try once
        return True

    def record_success(self):
        self._failures = 0
        self._state = "closed"
        self._last_failure_time = None

    def record_failure(self):
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.FAILURE_THRESHOLD:
            self._state = "open"

    @property
    def state(self) -> str:
        return self._state


# Global per-provider circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {
    "groq": CircuitBreaker(),
    "minimax": CircuitBreaker(),
    "openai": CircuitBreaker(),
}


def get_circuit_breaker(provider: str) -> CircuitBreaker:
    return _circuit_breakers.get(provider.lower(), CircuitBreaker())


# ─── Exceptions ─────────────────────────────────────────────────────────────

class LLMCircuitOpenError(Exception):
    """Raised when the circuit breaker is open (provider is down)."""
    pass


class LLMTimeoutError(Exception):
    """Raised when the LLM provider does not respond within the timeout."""
    pass


class LLMProviderError(Exception):
    """Raised when the LLM provider returns a non-retryable error."""
    pass


# ─── Direct LLM chat (for non-LangChain use) ───────────────────────────────

async def llm_chat(
    messages: list[dict],
    model: Optional[str] = None,
    provider: str = "groq",
    timeout: float = 60.0,
    temperature: float = 0.7,
) -> dict:
    """
    Call the LLM via the configured provider with timeout + circuit breaker.

    Returns the raw response dict from the provider.

    Raises:
        LLMCircuitOpenError: circuit breaker is open
        LLMTimeoutError: request timed out
        LLMProviderError: non-retryable provider error
    """
    import httpx
    import openai

    breaker = get_circuit_breaker(provider)

    if not breaker.should_try():
        raise LLMCircuitOpenError(
            f"LLM provider '{provider}' is currently unavailable (circuit open). Please try again later."
        )

    from app.config import settings

    # Select base URL + api key
    if provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = settings.groq_api_key
        model = model or settings.groq_model
    elif provider == "minimax":
        base_url = "https://api.minimax.chat/v1"
        api_key = settings.minimax_api_key
        model = model or settings.minimax_model
    else:
        base_url = settings.openai_api_base
        api_key = settings.openai_api_key
        model = model or "gpt-4o-mini"

    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=httpx.Timeout(timeout),
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        breaker.record_success()
        return response.model_dump()
    except openai.APITimeoutError:
        breaker.record_failure()
        raise LLMTimeoutError(f"LLM request timed out after {timeout}s")
    except openai.APIStatusError as e:
        breaker.record_failure()
        raise LLMProviderError(f"LLM provider error ({e.status_code}): {e.message}")
    except Exception as e:
        breaker.record_failure()
        raise LLMProviderError(f"Unexpected LLM error: {e}")


def check_circuit_breaker(provider: str) -> None:
    """Raise LLMCircuitOpenError if the circuit breaker for the given provider is open."""
    breaker = get_circuit_breaker(provider)
    if not breaker.should_try():
        raise LLMCircuitOpenError(
            f"LLM provider '{provider}' is currently unavailable (circuit open). Please try again later."
        )
