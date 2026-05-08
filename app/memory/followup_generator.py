"""
FollowUpGenerator — generates 2-3 suggested follow-up questions
based on the conversation context, using the LLM.
"""
import logging
from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
    def __init__(
        self,
        llm_provider,
        max_questions: int = 3,
        prompt: Optional[ChatPromptTemplate] = None,
        enabled: bool = True,
    ):
        self._llm_provider = llm_provider
        self.max_questions = max_questions
        self.prompt = prompt or DEFAULT_FOLLOWUP_PROMPT
        self.enabled = enabled

    def generate(
        self,
        conversation_history: str,
        current_answer: Optional[str] = None,
    ) -> List[str]:
        if not self.enabled:
            return []
        if not conversation_history.strip():
            return []

        try:
            llm = self._llm_provider.get_llm(temperature=0.3)
            chain = self.prompt | llm | JsonOutputParser()

            context = conversation_history
            if current_answer:
                context = f"{conversation_history}\n\nAssistant: {current_answer}"

            raw = chain.invoke({"conversation_history": context})
            questions = raw.get("questions", [])
            return [str(q).strip() for q in questions[: self.max_questions] if q]

        except Exception as exc:
            logger.warning("Follow-up generation failed: %s", exc)
            return []


def generate_followup_questions(
    llm_provider,
    conversation_history: str,
    current_answer: Optional[str] = None,
    followup_enabled: bool = True,
) -> List[str]:
    if not followup_enabled:
        return []
    gen = FollowUpGenerator(llm_provider, enabled=followup_enabled)
    return gen.generate(conversation_history, current_answer)
