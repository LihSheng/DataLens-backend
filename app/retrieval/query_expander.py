"""
QueryExpander — expands a user query into sub-questions using an LLM.
Improves recall by searching for related concepts in addition to the original query.
"""
import json
import logging
from typing import List, Optional

from app.services.vectorstore_service import get_llm

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands a user query into 2-3 related sub-questions using the LLM.
    The expanded queries are used together with the original to increase recall.

    Args:
        llm: Optional pre-configured LLM instance. If not provided, uses get_llm().
    """

    EXPANSION_PROMPT = """You are a query expansion assistant for a RAG system.
Given the user's query, generate 2-3 related sub-questions that would help retrieve
relevant documents. Return ONLY a JSON list of strings, nothing else.

Example:
Query: "What are the tax implications of remote work?"
Response: ["tax deduction for home office", "remote work expense reimbursement laws", "employment tax for remote employees"]

Query: "{query}"
Response:"""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def expand(self, query: str) -> List[str]:
        """
        Returns a list of expanded queries, always including the original query first.

        The LLM is asked to produce a JSON list of related sub-questions.
        If parsing fails, falls back to returning just the original query.
        """
        llm = self._get_llm()
        try:
            response = llm.invoke(self.EXPANSION_PROMPT.format(query=query))
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse as JSON list
            try:
                expanded = json.loads(content)
                if isinstance(expanded, list):
                    # Deduplicate: keep original + up to 2 unique expansions
                    unique = [q for q in expanded if q != query][:2]
                    return [query] + unique
            except json.JSONDecodeError:
                logger.warning("QueryExpander: LLM response was not valid JSON, using original query.")

            return [query]
        except Exception as e:
            logger.warning(f"QueryExpander: LLM invocation failed: {e}")
            return [query]

    def __call__(self, query: str) -> List[str]:
        """Convenience callable — mirrors expand()."""
        return self.expand(query)
