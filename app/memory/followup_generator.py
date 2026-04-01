"""
FollowUpGenerator — generates 2-3 suggested follow-up questions
based on the conversation context, using the LLM.
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.vectorstore_service import get_llm

logger = logging.getLogger(__name__)

DEFAULT_FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Based on the conversation history below, "
        "generate exactly 3 natural follow-up questions that the user might want to ask next.\n"
        "Each question should be concise (under 20 words), specific, and directly related to "
        "the previous exchange.\n"
        "Return a valid JSON object with a 'questions' key — a list of exactly 3 strings.\n"
        "Do not include greetings, apologies, or meta-commentary. Only the questions.",
    ),
    (
        "human",
        "Conversation:\n{conversation_history}\n\n"
        "Generate 3 follow-up questions as JSON.",
    ),
])


class FollowUpGenerator:
    """
    Generate suggested follow-up questions from conversation history.

    Args:
        max_questions: Number of questions to generate (default 3).
        prompt: Custom prompt template (optional).
        enabled: Master kill-switch.
    """

    def __init__(
        self,
        max_questions: int = 3,
        prompt: Optional[ChatPromptTemplate] = None,
        enabled: bool = True,
    ):
        self.max_questions = max_questions
        self.prompt = prompt or DEFAULT_FOLLOWUP_PROMPT
        self.enabled = enabled
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def generate(
        self,
        conversation_history: str,
        current_answer: Optional[str] = None,
    ) -> List[str]:
        """
        Generate follow-up questions.

        Args:
            conversation_history: Formatted prior messages (User: ... / Assistant: ...).
            current_answer: Optional answer to also consider for context.

        Returns:
            List of up to max_questions suggested question strings.
        """
        if not self.enabled:
            return []

        if not conversation_history.strip():
            return []

        try:
            llm = self._get_llm()
            chain = self.prompt | llm | JsonOutputParser()

            context = conversation_history
            if current_answer:
                context = f"{conversation_history}\n\nAssistant: {current_answer}"

            raw = chain.invoke({"conversation_history": context})
            questions = raw.get("questions", [])

            # Ensure we return a clean list of strings
            return [str(q).strip() for q in questions[: self.max_questions] if q]

        except Exception as exc:
            logger.warning("Follow-up generation failed: %s", exc)
            return []


def generate_followup_questions(
    conversation_history: str,
    current_answer: Optional[str] = None,
    followup_enabled: bool = True,
) -> List[str]:
    """
    Convenience function.
    """
    if not followup_enabled:
        return []

    gen = FollowUpGenerator(enabled=followup_enabled)
    return gen.generate(conversation_history, current_answer)
