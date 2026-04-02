"""
Reranker — re-ranks retrieved documents using a cross-encoder or Cohere Rerank API.
"""
import logging
from typing import List, Optional

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-ranks documents using a cross-encoder model or Cohere Rerank API.

    Preference order:
    1. Cohere Rerank API (if COHERE_API_KEY is set)
    2. Local cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

    Args:
        model_name: Override for the cross-encoder model name.
        cohere_api_key: Override for the Cohere API key.
    """

    def __init__(self, model_name: Optional[str] = None, cohere_api_key: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model
        self.cohere_api_key = cohere_api_key or settings.cohere_api_key
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Lazy-load the local cross-encoder model on first use."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local reranking. "
                    "Install with: pip install sentence-transformers"
                )
            self._cross_encoder = CrossEncoder(self.model_name)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """
        Returns documents re-ranked by relevance to the query.

        Args:
            query: The search query.
            documents: List of documents to re-rank.
            top_n: Number of top documents to return after re-ranking.

        Returns:
            Re-ranked list of documents (most relevant first).
        """
        if not documents:
            return []

        top_n = top_n or len(documents)

        if self.cohere_api_key:
            return self._rerank_cohere(query, documents, top_n)
        else:
            return self._rerank_cross_encoder(query, documents, top_n)

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
    ) -> List[Document]:
        """Re-rank using a local sentence-transformers cross-encoder."""
        model = self._get_cross_encoder()
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs)
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_n]:
            doc.metadata["reranker_score"] = float(score)
            results.append(doc)
        return results

    def _rerank_cohere(
        self,
        query: str,
        documents: List[Document],
        top_n: int,
    ) -> List[Document]:
        """Re-rank using the Cohere Rerank API."""
        import httpx

        headers = {"Authorization": f"Bearer {self.cohere_api_key}"}
        payload = {
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_n,
            "model": "rerank-english-v2.0",
        }
        with httpx.Client() as client:
            resp = client.post(
                "https://api.cohere.ai/v1/rerank",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return [documents[r["index"]] for r in results]

    def __call__(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """Convenience callable — mirrors rerank()."""
        return self.rerank(query, documents, top_n)
