"""
LLMProvider — deep module for LLM backend selection with circuit breaker.

Hides provider-specific API keys, model names, base URLs, and circuit breaker
state behind a single interface. Adapters return LangChain-compatible chat models.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker (internal to the LLMProvider module)
# ---------------------------------------------------------------------------


class CircuitBreaker:
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 30

    def __init__(self) -> None:
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"

    def should_try(self) -> bool:
        now = time.time()
        if self._state == "closed":
            return True
        if self._state == "open":
            if self._last_failure_time and (now - self._last_failure_time) >= self.RECOVERY_TIMEOUT:
                self._state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"
        self._last_failure_time = None

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.FAILURE_THRESHOLD:
            self._state = "open"

    @property
    def state(self) -> str:
        return self._state


class CircuitOpenError(Exception):
    pass


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    @abstractmethod
    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> BaseChatModel:
        """Return a LangChain-compatible chat model with circuit-breaker guard."""

    @abstractmethod
    def ensure_circuit_ok(self) -> None:
        """Raise CircuitOpenError if the circuit breaker forbids calls."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """The default model name for this provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (groq, minimax, openai)."""


# ---------------------------------------------------------------------------
# Groq adapter
# ---------------------------------------------------------------------------


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._breaker = CircuitBreaker()

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "groq"

    def ensure_circuit_ok(self) -> None:
        if not self._breaker.should_try():
            raise CircuitOpenError(
                "LLM provider 'groq' is currently unavailable (circuit open)."
            )

    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> BaseChatModel:
        self.ensure_circuit_ok()
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model or self._model,
            api_key=self._api_key,
            temperature=temperature,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# MiniMax / OpenAI-compatible adapter
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.chat/v1",
        model: str = "MiniMax-Text-01",
        provider_name: str = "minimax",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._provider_name = provider_name
        self._breaker = CircuitBreaker()

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def ensure_circuit_ok(self) -> None:
        if not self._breaker.should_try():
            raise CircuitOpenError(
                f"LLM provider '{self._provider_name}' is currently unavailable (circuit open)."
            )

    def get_llm(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> BaseChatModel:
        self.ensure_circuit_ok()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or self._model,
            openai_api_key=self._api_key,
            openai_api_base=self._base_url,
            temperature=temperature,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_provider(
    provider: str = "groq",
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    minimax_api_key: str = "",
    minimax_model: str = "MiniMax-Text-01",
    openai_api_key: str = "",
    openai_api_base: str = "https://api.openai.com/v1",
) -> LLMProvider:
    if provider == "groq":
        return GroqProvider(api_key=groq_api_key, model=groq_model)
    if provider == "minimax":
        return OpenAICompatibleProvider(
            api_key=minimax_api_key,
            base_url="https://api.minimax.chat/v1",
            model=minimax_model,
            provider_name="minimax",
        )
    return OpenAICompatibleProvider(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model=minimax_model or "gpt-4o-mini",
        provider_name="openai",
    )
