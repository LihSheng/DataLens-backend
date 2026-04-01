"""
BM25Retriever — sparse retriever using BM25 (rank_bm25).
Complements dense (vector) retrieval with keyword-matched ranking.
"""
import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    import rank_bm25
except ImportError:
    rank_bm25 = None


class BM25Retriever:
    """
    Sparse retriever backed by BM25 — indexes raw text chunks and ranks
    them by keyword overlap with the query.

    Args:
        texts: List of raw text strings to index.
        metadata: List of metadata dicts aligned with texts.
        k: Default number of top results to return.
    """

    def __init__(self, texts: List[str], metadata: List[dict], k: int = 20):
        if rank_bm25 is None:
            raise ImportError(
                "rank_bm25 is required for BM25Retriever. "
                "Install with: pip install rank-bm25"
            )
        self.texts = texts
        self.metadata = metadata
        self.k = k
        self._tokenized = [t.split(" ") for t in texts]
        self._bm25 = rank_bm25.BM25Okapi(self._tokenized)

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Return the top-k documents most relevant to the query by BM25 score.
        """
        k = k or self.k
        scores = self._bm25.get_scores(query.split(" "))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Document(page_content=self.texts[i], metadata=self.metadata[i])
            for i in top_indices
        ]

    def invoke(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Alias for get_relevant_documents — conforms to LangChain Runnable interface."""
        return self.get_relevant_documents(query, k)
