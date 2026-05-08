"""
QueryExpander — expands a user query into sub-questions using an LLM.
Improves recall by searching for related concepts in addition to the original query.
"""
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    EXPANSION_PROMPT = """You are a query expansion assistant for a RAG system.
Given the user's query, generate 2-3 related sub-questions that would help retrieve
relevant documents. Return ONLY a JSON list of strings, nothing else.

Example:
Query: "What are the tax implications of remote work?"
Response: ["tax deduction for home office", "remote work expense reimbursement laws", "employment tax for remote employees"]

Query: "{query}"
Response:"""

    def __init__(self, llm):
        self._llm = llm

    def expand(self, query: str) -> List[str]:
        llm = self._llm.get_llm()
        try:
            response = llm.invoke(self.EXPANSION_PROMPT.format(query=query))
            content = response.content if hasattr(response, "content") else str(response)

            try:
                expanded = json.loads(content)
                if isinstance(expanded, list):
                    unique = [q for q in expanded if q != query][:2]
                    return [query] + unique
            except json.JSONDecodeError:
                logger.warning("QueryExpander: LLM response was not valid JSON, using original query.")

            return [query]
        except Exception as e:
            logger.warning(f"QueryExpander: LLM invocation failed: {e}")
            return [query]

    def __call__(self, query: str) -> List[str]:
        return self.expand(query)
