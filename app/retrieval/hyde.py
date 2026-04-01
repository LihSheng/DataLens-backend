"""
HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer
and embed it for better retrieval.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
"""
import logging
from typing import List, Optional

from langchain_core.documents import Document

from app.services.vectorstore_service import get_llm

logger = logging.getLogger(__name__)


class HyDE:
    """
    HyDE generates a hypothetical answer document from the query, then retrieves
    based on that hypothetical document's embedding rather than the query itself.

    Process:
      1. LLM generates a short hypothetical answer (2-3 sentences)
      2. Hypothetical answer is embedded with the same embedding model as real docs
      3. Hypothetical embedding is used for similarity search in the vectorstore

    This approach often improves recall because the hypothetical answer
    shares lexical and semantic overlap with actual indexed documents.

    Args:
        llm: Optional pre-configured LLM instance.
        embedding: Optional pre-configured embedding function.
    """

    HYDE_PROMPT = """You are a helpful research assistant. Given the user's question,
write a brief hypothetical answer (2-3 sentences) that would answer the question
as if you knew the relevant context from the documents.

Question: {query}
Hypothetical Answer:"""

    def __init__(self, llm=None, embedding=None):
        self._llm = llm
        self._embedding = embedding

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _get_embedding(self):
        if self._embedding is None:
            from app.services.vectorstore_service import embeddings as emb

            self._embedding = emb
        return self._embedding

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical answer that would answer the query.

        Returns:
            A string containing the hypothetical answer text.
        """
        llm = self._get_llm()
        prompt = self.HYDE_PROMPT.format(query=query)
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"HyDE: LLM hypothetical document generation failed: {e}")
            return query  # Fallback to using the original query as a last resort

    def embed_hypothetical(self, hypothetical_text: str) -> List[float]:
        """
        Embed the hypothetical document using the configured embedding model.

        Returns:
            A list of floats representing the embedding vector.
        """
        emb = self._get_embedding()
        return emb.embed_query(hypothetical_text)

    def retrieve_with_hypothetical(
        self,
        query: str,
        vectorstore,
        k: int = 4,
    ) -> List[Document]:
        """
        Full HyDE retrieval pipeline:
          1. Generate hypothetical answer
          2. Embed it
          3. Search vectorstore with the hypothetical embedding

        Args:
            query: The user's original question.
            vectorstore: A FAISS (or compatible) vectorstore with similarity_search_by_vector.
            k: Number of documents to retrieve.

        Returns:
            List of retrieved Documents.
        """
        hypothetical = self.generate_hypothetical_document(query)
        emb = self._get_embedding()
        query_embedding = emb.embed_query(hypothetical)
        try:
            docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)
        except Exception as e:
            logger.warning(f"HyDE: vectorstore similarity search failed: {e}, falling back to query embedding")
            docs = vectorstore.similarity_search(query, k=k)
        return docs

    def __call__(self, query: str, vectorstore, k: int = 4) -> List[Document]:
        """Convenience callable — mirrors retrieve_with_hypothetical()."""
        return self.retrieve_with_hypothetical(query, vectorstore, k)
