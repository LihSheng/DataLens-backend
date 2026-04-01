"""Simple model router for latency/cost/quality trade-offs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import settings as app_settings


@dataclass
class RouteDecision:
    model: str
    reason: str


class ModelRouter:
    """
    Rule-based model router.

    Modes:
      - balanced: prefer speed for simple asks, quality for complex/context-heavy asks
      - cost: always prefer fast model
      - quality: always prefer quality model
    """

    def __init__(
        self,
        default_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        quality_model: Optional[str] = None,
    ):
        provider_default = (
            app_settings.groq_model
            if app_settings.use_provider == "groq"
            else app_settings.minimax_model
        )
        self.default_model = default_model or provider_default
        self.fast_model = fast_model or self.default_model
        self.quality_model = quality_model or self.default_model

    def route(
        self,
        *,
        question: str,
        context_tokens: int,
        settings: Optional[Dict[str, Any]] = None,
    ) -> RouteDecision:
        settings = settings or {}
        if settings.get("model"):
            return RouteDecision(
                model=str(settings["model"]),
                reason="explicit model override",
            )

        mode = str(settings.get("routing_mode", "balanced")).lower()
        fast_model = str(settings.get("fast_model") or self.fast_model)
        quality_model = str(settings.get("quality_model") or self.quality_model)

        if mode == "cost":
            return RouteDecision(model=fast_model, reason="cost mode")
        if mode == "quality":
            return RouteDecision(model=quality_model, reason="quality mode")

        question_len = len(question.split())
        if context_tokens >= 1400 or question_len >= 35:
            return RouteDecision(model=quality_model, reason="complex query/context")
        return RouteDecision(model=fast_model, reason="simple query")
