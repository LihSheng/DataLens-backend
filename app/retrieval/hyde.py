"""
HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer
and embed it for better retrieval.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
"""
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HyDE:
    HYDE_PROMPT = """You are a helpful research assistant. Given the user's question,
write a brief hypothetical answer (2-3 sentences) that would answer the question
as if you knew the relevant context from the documents.

Question: {query}
Hypothetical Answer:"""

    def __init__(self, llm_provider):
        self._llm_provider = llm_provider

    def generate_hypothetical_document(self, query: str) -> str:
        llm = self._llm_provider.get_llm()
        prompt = self.HYDE_PROMPT.format(query=query)
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"HyDE: LLM hypothetical document generation failed: {e}")
            return query

    def __call__(self, query: str, vectorstore, k: int = 4) -> List[Document]:
        hypothetical = self.generate_hypothetical_document(query)
        emb = vectorstore.embed_query(hypothetical)
        try:
            return vectorstore.search_by_vector(emb, k=k)
        except Exception as e:
            logger.warning(f"HyDE: vectorstore search failed: {e}, falling back")
            return vectorstore.search(query, k=k)
