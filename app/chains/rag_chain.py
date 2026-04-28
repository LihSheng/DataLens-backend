"""
RAG chain with retrieval, safety, memory, and Stage 5 performance features.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry import trace as otel_trace

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.cache.semantic_cache import SemanticCache
from app.context.assembler import ContextAssembler
from app.memory.conversation_memory import get_conversation_memory
from app.memory.followup_generator import generate_followup_questions
from app.retrieval.filters import apply_filters
from app.retrieval.hyde import HyDE
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_expander import QueryExpander
from app.retrieval.reranker import Reranker
from app.routing.model_router import ModelRouter
from app.safety.safety_layer import SafetyLayer, SafetyResponse
from app.services.cost_tracker import estimate_cost_usd
from app.services.vectorstore_service import get_llm, get_vectorstore
from app.services.llm_runner import check_circuit_breaker, LLMCircuitOpenError
from app.config import settings as app_settings

logger = logging.getLogger(__name__)

tracer = otel_trace.get_tracer(__name__)

def _safe_print(text: str) -> None:
    """
    Windows PowerShell often uses a legacy console encoding (e.g. cp1252).
    Printing prompts with Unicode can crash request handling via UnicodeEncodeError.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)

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
    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 8,
        rerank_top_n: int = 4,
        conversation_id: Optional[str] = None,
        conversation_history_limit: int = 10,
        enable_memory: bool = True,
        enable_followup: bool = True,
        trace_id: Optional[str] = None,
    ):
        # Merge runtime chain settings with env-backed defaults.
        base_settings: Dict[str, Any] = {
            "semantic_cache_enabled": bool(app_settings.semantic_cache_enabled),
            "semantic_cache_threshold": float(app_settings.semantic_cache_threshold),
            "context_max_tokens": int(app_settings.context_max_tokens),
            "routing_mode": str(app_settings.routing_mode),
            "fast_model": str(app_settings.fast_model),
            "quality_model": str(app_settings.quality_model),
        }
        self.settings = dict(base_settings)
        if settings:
            self.settings.update(settings)
        self.filters = filters or {}
        self.k = k
        self.rerank_top_n = rerank_top_n
        self.conversation_id = conversation_id
        self.conversation_history_limit = conversation_history_limit
        self.enable_memory = enable_memory
        self.enable_followup = enable_followup
        # Extract trace_id from OTel context if not provided
        if trace_id:
            self.trace_id = trace_id
        else:
            ctx = otel_trace.get_current_span().get_span_context()
            self.trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else ""

        self._retriever = None
        self._reranker = None
        self._query_expander = None
        self._hyde = None
        self._vectorstore = None
        self._safety_layer: Optional[SafetyLayer] = None
        self._conversation_memory = None
        self._semantic_cache: Optional[SemanticCache] = None
        self._context_assembler: Optional[ContextAssembler] = None
        self._model_router: Optional[ModelRouter] = None
        self._last_run_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Memory + safety
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

    def _get_safety_layer(self) -> SafetyLayer:
        if self._safety_layer is None:
            self._safety_layer = SafetyLayer(settings=self.settings)
        return self._safety_layer

    # ------------------------------------------------------------------
    # Components (Stage 5)
    # ------------------------------------------------------------------

    def _get_semantic_cache(self) -> SemanticCache:
        if self._semantic_cache is None:
            self._semantic_cache = SemanticCache(
                similarity_threshold=float(
                    self.settings.get("semantic_cache_threshold", 0.9)
                )
            )
        return self._semantic_cache

    def _get_context_assembler(self) -> ContextAssembler:
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler(
                max_context_tokens=int(self.settings.get("context_max_tokens", 1800)),
            )
        return self._context_assembler

    def _get_model_router(self) -> ModelRouter:
        if self._model_router is None:
            self._model_router = ModelRouter()
        return self._model_router

    # ------------------------------------------------------------------
    # LLM + vectorstore accessors
    # ------------------------------------------------------------------

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
        return vs.as_retriever(search_kwargs={"k": self.k})

    def _retrieve(self, query: str) -> List[Document]:
        use_expansion = self.settings.get("query_expansion", False)
        use_hyde = self.settings.get("hyde", False)

        with tracer.start_as_current_span("retrieval") as span:
            span.set_attribute("retrieval.raw_chunks", 0)
            span.set_attribute("retrieval.filters_applied", bool(self.filters))
            span.set_attribute("retrieval.query_expansion", use_expansion)
            span.set_attribute("retrieval.hyde", use_hyde)

            queries = [query]
            if use_expansion:
                queries = self._get_query_expander().expand(query)

            all_docs: List[Document] = []
            vs = self._get_vectorstore()
            for q in queries:
                if use_hyde:
                    docs = self._get_hyde()(q, vs, k=self.k)
                else:
                    docs = self._build_retriever().invoke(q, k=self.k)
                all_docs.extend(docs)

            seen = set()
            unique_docs = []
            for doc in all_docs:
                key = doc.page_content[:120]
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)

            if self.filters:
                unique_docs = apply_filters(unique_docs, **self.filters)

            span.set_attribute("retrieval.raw_chunks", len(unique_docs))
            return unique_docs

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        with tracer.start_as_current_span("cross_encoder_rerank") as span:
            span.set_attribute("reranker.input_docs", len(docs))
            if not self.settings.get("reranker", False):
                span.set_attribute("reranker.model", "disabled")
                span.set_attribute("reranker.output_docs", len(docs[: self.rerank_top_n]))
                return docs[: self.rerank_top_n]

            reranker = self._get_reranker()
            span.set_attribute("reranker.model", getattr(reranker, "model_name", "unknown"))
            output_docs = reranker.rerank(query, docs, top_n=self.rerank_top_n)
            span.set_attribute("reranker.output_docs", len(output_docs))
            if output_docs:
                top_score = output_docs[0].metadata.get("reranker_score")
                if top_score is not None:
                    span.set_attribute("reranker.top_score", top_score)
            return output_docs

    # ------------------------------------------------------------------
    # Stage 5 helpers
    # ------------------------------------------------------------------

    def _cache_namespace(self) -> str:
        namespace = {
            "filters": self.filters,
            "query_expansion": self.settings.get("query_expansion", False),
            "hyde": self.settings.get("hyde", False),
            "reranker": self.settings.get("reranker", False),
            "memory": self.enable_memory and bool(self.conversation_id),
        }
        if namespace["memory"]:
            namespace["conversation_id"] = self.conversation_id
        return json.dumps(namespace, sort_keys=True)

    @staticmethod
    def _serialize_docs(docs: List[Document]) -> List[Dict[str, Any]]:
        return [
            {"page_content": d.page_content, "metadata": d.metadata or {}}
            for d in docs
        ]

    @staticmethod
    def _deserialize_docs(items: List[Dict[str, Any]]) -> List[Document]:
        return [
            Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}))
            for item in items
        ]

    def _compute_prompt_tokens(
        self,
        *,
        prompt_template: ChatPromptTemplate,
        prompt_values: Dict[str, Any],
        answer: str,
    ) -> Tuple[int, int]:
        asm = self._get_context_assembler()
        prompt_value = prompt_template.invoke(prompt_values)
        raw_prompt = "\n".join(
            str(getattr(msg, "content", ""))
            for msg in prompt_value.to_messages()
        )
        input_tokens = asm.count_tokens(raw_prompt)
        output_tokens = asm.count_tokens(answer)
        return input_tokens, output_tokens

    def _build_prompt(self, include_history: bool) -> ChatPromptTemplate:
        system_prompt = self.settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        if include_history:
            user_prompt = self.settings.get("user_prompt", DEFAULT_USER_PROMPT_WITH_MEMORY)
        else:
            user_prompt = self.settings.get("user_prompt", DEFAULT_USER_PROMPT)
        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", user_prompt)]
        )

    def _run_chain(self, question: str) -> Tuple[str, List[Document]]:
        history = self._get_conversation_history() if self.enable_memory else ""

        cache_enabled = bool(self.settings.get("semantic_cache_enabled", True))
        namespace = self._cache_namespace()

        if cache_enabled:
            hit = self._get_semantic_cache().get(query=question, namespace=namespace)
            if hit is not None:
                docs = self._deserialize_docs(hit.sources)
                include_history = bool(history and self.enable_memory)
                prompt = self._build_prompt(include_history=include_history)
                prompt_values = {"question": question, "context": "cached_response"}
                if include_history:
                    prompt_values["conversation_history"] = history
                input_tokens, output_tokens = self._compute_prompt_tokens(
                    prompt_template=prompt,
                    prompt_values=prompt_values,
                    answer=hit.answer,
                )
                model = hit.model or self._get_model_router().default_model
                self._last_run_meta = {
                    "cache_hit": True,
                    "cache_similarity": hit.similarity,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": estimate_cost_usd(model, input_tokens, output_tokens),
                }
                return hit.answer, docs

        docs = self._retrieve(question)
        reranked = self._rerank(question, docs)
        with tracer.start_as_current_span("context_assembly") as span:
            assembly = self._get_context_assembler().assemble(
                reranked,
                question=question,
                max_context_tokens=self.settings.get("context_max_tokens"),
            )
            span.set_attribute("context.chunks_used", len(assembly.selected_docs))
            span.set_attribute("context.tokens_used", assembly.context_tokens)
            span.set_attribute(
                "context.token_budget",
                self.settings.get("context_max_tokens") or 1800,
            )

        route = self._get_model_router().route(
            question=question,
            context_tokens=assembly.context_tokens,
            settings=self.settings,
        )
        llm = get_llm(model_name=route.model)

        include_history = bool(history and self.enable_memory)
        prompt = self._build_prompt(include_history=include_history)
        chain = prompt | llm | StrOutputParser()

        payload = {"question": question, "context": assembly.context}
        if include_history:
            payload["conversation_history"] = history

        # Debug logging must never dump full context/prompt in normal runs (PII risk).
        if app_settings.log_llm_prompt:
            _safe_print(
                "[LLM PROMPT] "
                f"model={route.model} "
                f"question_chars={len(question or '')} "
                f"context_chars={len(assembly.context or '')} "
                f"history_chars={len(history or '')} "
                f"chunks={len(assembly.selected_docs)}"
            )

        # Circuit breaker — fail fast if provider is known down
        check_circuit_breaker(app_settings.use_provider)

        answer = chain.invoke(payload)

        input_tokens, output_tokens = self._compute_prompt_tokens(
            prompt_template=prompt,
            prompt_values=payload,
            answer=answer,
        )
        self._last_run_meta = {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "model": route.model,
            "route_reason": route.reason,
            "context_tokens": assembly.context_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": estimate_cost_usd(route.model, input_tokens, output_tokens),
        }

        if cache_enabled:
            self._get_semantic_cache().set(
                query=question,
                answer=answer,
                sources=self._serialize_docs(assembly.selected_docs),
                namespace=namespace,
                model=route.model,
            )

        return answer, assembly.selected_docs

    # ------------------------------------------------------------------
    # Public invoke
    # ------------------------------------------------------------------

    def invoke(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        question = input_.get("question") or input_.get("query")

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
                **self._last_run_meta,
            }

        answer, docs = self._run_chain(question)
        return {
            "answer": answer,
            "source_documents": docs,
            "fallback_triggered": False,
            "trace_id": self.trace_id,
            "followup_questions": self._generate_followups(answer),
            **self._last_run_meta,
        }

    def _generate_followups(self, answer: str) -> List[str]:
        if not self.enable_followup:
            return []
        history = self._get_conversation_history()
        return generate_followup_questions(
            conversation_history=history,
            current_answer=answer,
            followup_enabled=self.enable_followup,
        )

    def __call__(self, input_: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input_)

