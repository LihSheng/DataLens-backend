"""
RAG chain — ties together retrieval components with the LLM.

Integrates:
  - HybridRetriever (dense FAISS + sparse BM25 with RRF)
  - Optional QueryExpander (LLM-generated sub-questions)
  - Optional HyDE (hypothetical document embeddings)
  - Optional Reranker (cross-encoder or Cohere)
  - SafetyLayer (Stage 3): guardrails, prompt injection, grounding, citations
  - ConversationMemory (Stage 4): Redis-backed conversation history injection
  - FollowUpGenerator (Stage 4): suggested follow-up questions
  - trace_id propagation throughout

Backward-compatible with the existing /api/chat endpoint contract.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.memory.conversation_memory import get_conversation_memory
from app.memory.followup_generator import generate_followup_questions
from app.retrieval.filters import apply_filters
from app.retrieval.hyde import HyDE
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_expander import QueryExpander
from app.retrieval.reranker import Reranker
from app.safety.safety_layer import SafetyLayer, SafetyResponse
from app.services.vectorstore_service import get_llm, get_vectorstore

logger = logging.getLogger(__name__)

# --- Default prompt templates ---
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant answering questions based on the provided context.\n"
    "If the context does not contain enough information to answer the question, say so clearly.\n"
    "Always cite relevant parts of the context when answering."
)

DEFAULT_USER_PROMPT_WITH_MEMORY = (
    "Conversation history:\n"
    "{conversation_history}\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Answer based strictly on the provided context."
)

DEFAULT_USER_PROMPT = (
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Answer based strictly on the provided context."
)


class RAGChain:
    """
    Full RAG chain with a configurable retrieval pipeline.

    Args:
        settings: Dict of flags / options:
            - query_expansion: bool  — expand query with LLM sub-questions
            - hyde: bool             — use hypothetical document embeddings
            - reranker: bool         — re-rank results with cross-encoder / Cohere
            - confidence_threshold: float
            - system_prompt: str
            - user_prompt: str
            # Safety (Stage 3)
            - guardrails_enabled: bool
            - prompt_injection_check: bool
            - grounding_check: bool
            - citation_validation: bool
            - grounding_threshold: float
            - injection_confidence_threshold: float
            - required_citation_threshold: float
            - max_retries: int
        filters: Optional dict of filters applied at retrieval time:
            - document_ids: List[str]
            - sources: List[str]
            - tags: List[str]
        k: Number of documents to retrieve per query (before re-ranking).
        rerank_top_n: Number of documents to return after re-ranking.
    """

    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 8,
        rerank_top_n: int = 4,
        # Stage 4 — memory
        conversation_id: Optional[str] = None,
        conversation_history_limit: int = 10,
        enable_memory: bool = True,
        enable_followup: bool = True,
        trace_id: Optional[str] = None,
    ):
        self.settings = settings or {}
        self.filters = filters or {}
        self.k = k
        self.rerank_top_n = rerank_top_n
        self.conversation_id = conversation_id
        self.conversation_history_limit = conversation_history_limit
        self.enable_memory = enable_memory
        self.enable_followup = enable_followup
        self.trace_id = trace_id or str(uuid.uuid4())

        self._retriever = None
        self._reranker = None
        self._query_expander = None
        self._hyde = None
        self._llm = None
        self._vectorstore = None
        self._chain = None
        self._safety_layer: Optional[SafetyLayer] = None
        self._conversation_memory = None

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------

    def _get_conversation_memory(self):
        if self._conversation_memory is None:
            self._conversation_memory = get_conversation_memory()
        return self._conversation_memory

    def _get_conversation_history(self) -> str:
        if not self.conversation_id:
            return ""
        mem = self._get_conversation_memory()
        return mem.get_formatted_history(
            self.conversation_id,
            limit=self.conversation_history_limit,
        )

    # ------------------------------------------------------------------
    # Safety layer
    # ------------------------------------------------------------------

    def _get_safety_layer(self) -> SafetyLayer:
        if self._safety_layer is None:
            self._safety_layer = SafetyLayer(settings=self.settings)
        return self._safety_layer

    # ------------------------------------------------------------------
    # LLM + vectorstore accessors (lazy)
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _get_vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore()
        return self._vectorstore

    def _get_query_expander(self) -> QueryExpander:
        if self._query_expander is None:
            self._query_expander = QueryExpander()
        return self._query_expander

    def _get_hyde(self) -> HyDE:
        if self._hyde is None:
            self._hyde = HyDE()
        return self._hyde

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------

    def _build_retriever(self):
        """
        Build a retriever from the vectorstore.

        Returns:
            HybridRetriever if BM25 texts are available (from ingestion),
            otherwise falls back to plain FAISS similarity search retriever.
        """
        vs = self._get_vectorstore()
        bm25_texts = getattr(vs, "_bm25_texts", None)
        bm25_metadata = getattr(vs, "_bm25_metadata", None)

        if bm25_texts:
            return HybridRetriever(
                bm25_texts=bm25_texts,
                bm25_metadata=bm25_metadata,
                vectorstore=vs,
                k=self.k,
            )

        # Fallback: plain FAISS retriever
        return vs.as_retriever(search_kwargs={"k": self.k})

    def _retrieve(self, query: str) -> List[Document]:
        """
        Run the retrieval pipeline with optional expansion and HyDE.

        Multiple queries (from expansion) each return up to k documents.
        Results are deduplicated and filtered before being returned.
        """
        use_expansion = self.settings.get("query_expansion", False)
        use_hyde = self.settings.get("hyde", False)

        queries = [query]
        if use_expansion:
            expander = self._get_query_expander()
            queries = expander.expand(query)

        all_docs: List[Document] = []
        vs = self._get_vectorstore()

        for q in queries:
            if use_hyde:
                hyde = self._get_hyde()
                docs = hyde(q, vs, k=self.k)
            else:
                retriever = self._build_retriever()
                docs = retriever.invoke(q, k=self.k)
            all_docs.extend(docs)

        # Deduplicate by first 100 chars of page_content
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # Apply metadata filters
        if self.filters:
            unique_docs = apply_filters(unique_docs, **self.filters)

        return unique_docs

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Re-rank documents if the reranker is enabled.
        Otherwise, just return the top_n documents.
        """
        if not self.settings.get("reranker", False):
            return docs[: self.rerank_top_n]

        reranker = self._get_reranker()
        return reranker.rerank(query, docs, top_n=self.rerank_top_n)

    def _format_context(self, docs: List[Document]) -> str:
        """Format a list of documents into a context string for the LLM."""
        if not docs:
            return "No relevant context found."
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def _build_chain(self):
        """Assemble the LangChain Runnable chain."""
        llm = self._get_llm()
        system_prompt = self.settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        # Use memory-aware prompt if conversation history is available
        history = self._get_conversation_history()
        if history and self.enable_memory:
            user_prompt = self.settings.get(
                "user_prompt", DEFAULT_USER_PROMPT_WITH_MEMORY
            )
        else:
            user_prompt = self.settings.get("user_prompt", DEFAULT_USER_PROMPT)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        def retrieve_and_format(query: str) -> str:
            docs = self._retrieve(query)
            reranked = self._rerank(query, docs)
            return self._format_context(reranked)

        self._chain = (
            RunnablePassthrough.assign(context=retrieve_and_format)
            | prompt
            | llm
            | StrOutputParser()
        )

    def _run_chain(self, question: str) -> Tuple[str, List[Document]]:
        """
        Run the core RAG chain, returning (answer, source_documents).
        """
        if self._chain is None:
            self._build_chain()

        history = self._get_conversation_history() if self.enable_memory else ""
        chain_input = {"question": question, "query": question}
        if history and self.enable_memory:
            chain_input["conversation_history"] = history

        result = self._chain.invoke(chain_input)

        docs = self._retrieve(question)
        reranked = self._rerank(question, docs)

        return result, reranked

    # ------------------------------------------------------------------
    # Public invoke
    # ------------------------------------------------------------------

    def invoke(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full RAG chain with optional SafetyLayer checks.

        Args:
            input_: Dict containing at least one of:
                - "question": str
                - "query": str

        Returns:
            Dict with:
                - "answer": str  — the LLM's generated answer
                - "source_documents": List[Document]  — retrieved & re-ranked docs
                - "safety_response": SafetyResponse  — Stage 3 safety metadata
                - "fallback_triggered": bool
        """
        question = input_.get("question") or input_.get("query")

        # Check if safety features are enabled via any non-None safety flag
        safety_enabled = any(
            self.settings.get(flag) is not None
            for flag in (
                "guardrails_enabled",
                "prompt_injection_check",
                "grounding_check",
                "citation_validation",
            )
        )

        if safety_enabled:
            safety = self._get_safety_layer()

            def chain_fn(q: str) -> Tuple[str, List[Document]]:
                return self._run_chain(q)

            safety_response: SafetyResponse = safety.invoke(question, chain_fn)

            return {
                "answer": safety_response.answer,
                "source_documents": safety_response.source_documents,
                "safety_response": safety_response,
                "fallback_triggered": safety_response.fallback_triggered,
                "trace_id": self.trace_id,
                "followup_questions": self._generate_followups(safety_response.answer),
            }

        # Legacy path — no safety layer
        answer, docs = self._run_chain(question)
        return {
            "answer": answer,
            "source_documents": docs,
            "fallback_triggered": False,
            "trace_id": self.trace_id,
            "followup_questions": self._generate_followups(answer),
        }

    def _generate_followups(self, answer: str) -> List[str]:
        """Generate follow-up questions if enabled."""
        if not self.enable_followup:
            return []
        history = self._get_conversation_history()
        return generate_followup_questions(
            conversation_history=history,
            current_answer=answer,
            followup_enabled=self.enable_followup,
        )

    def __call__(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience callable — mirrors invoke()."""
        return self.invoke(input_)
