"""
HybridRetriever — dense (FAISS) + sparse (BM25) retrieval with RRF merging.
"""
import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.services.vectorstore_service import get_vectorstore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense (vector) + sparse (BM25) retrieval using
    Reciprocal Rank Fusion (RRF) for final ranking.

    RRF score = sum(1 / (k + rank)) where k=60 is a standard constant.

    Args:
        bm25_texts: Raw text strings for BM25 indexing.
        bm25_metadata: Metadata list aligned with bm25_texts.
        vectorstore: Optional pre-built FAISS vectorstore instance.
        k: Number of docs to fetch from each retriever before fusion.
        rrf_k: Constant k used in RRF formula (default 60).
    """

    def __init__(
        self,
        bm25_texts: Optional[List[str]] = None,
        bm25_metadata: Optional[List[dict]] = None,
        vectorstore=None,
        k: int = 20,
        rrf_k: int = 60,
    ):
        self._bm25 = None
        if bm25_texts and bm25_metadata:
            from app.retrieval.bm25_retriever import BM25Retriever

            self._bm25 = BM25Retriever(bm25_texts, bm25_metadata, k=k)
        self._vectorstore = vectorstore
        self.k = k
        self.rrf_k = rrf_k

    def _dense_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Dense vector search via FAISS. Returns list of (doc, score) tuples."""
        vs = self._vectorstore or get_vectorstore()
        docs = vs.similarity_search_with_score(query, k=k)
        return docs  # List[(Document, score)]

    def _sparse_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Sparse BM25 search. Returns list of (doc, score) tuples."""
        if self._bm25 is None:
            return []
        docs = self._bm25.invoke(query, k=k)
        # BM25 doesn't return scores by default; use rank order as pseudo-score
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

    @staticmethod
    def _rrf_fusion(
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion — merge two ranked lists into a single ranked list.

        Each retriever contributes a score of 1/(k+rank) for each document it returns.
        Scores are summed across retrievers; documents with higher aggregate scores rank higher.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc.page_content[:50] + str(doc.metadata)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = doc.page_content[:50] + str(doc.metadata)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Run hybrid retrieval and return fused ranked documents."""
        k = k or self.k
        dense = self._dense_search(query, k)
        sparse = self._sparse_search(query, k)
        return self._rrf_fusion(dense, sparse, k=self.rrf_k)

    def invoke(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Alias for get_relevant_documents — conforms to LangChain Runnable interface."""
        return self.get_relevant_documents(query, k)
