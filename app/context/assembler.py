"""Context assembler with token-budget controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken
from langchain_core.documents import Document


@dataclass
class ContextAssemblyResult:
    context: str
    selected_docs: List[Document]
    context_tokens: int


class ContextAssembler:
    """
    Build prompt context from retrieved documents while staying within a token budget.
    """

    def __init__(
        self,
        max_context_tokens: int = 1800,
        reserve_for_answer: int = 500,
        model_for_tokenizer: str = "gpt-4o-mini",
    ):
        self.max_context_tokens = max_context_tokens
        self.reserve_for_answer = reserve_for_answer
        try:
            self._encoding = tiktoken.encoding_for_model(model_for_tokenizer)
        except Exception:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text or ""))

    def _format_doc(self, doc: Document) -> str:
        src = doc.metadata.get("source", "unknown")
        return f"[Source: {src}]\n{doc.page_content}"

    def assemble(
        self,
        docs: List[Document],
        question: str = "",
        max_context_tokens: int | None = None,
    ) -> ContextAssemblyResult:
        if not docs:
            return ContextAssemblyResult(
                context="No relevant context found.",
                selected_docs=[],
                context_tokens=0,
            )

        budget = max_context_tokens or self.max_context_tokens
        budget = max(64, budget - self.reserve_for_answer)

        assembled_chunks: List[str] = []
        selected: List[Document] = []
        used_tokens = 0

        question_overhead = self.count_tokens(question) if question else 0
        budget = max(64, budget - question_overhead)

        for doc in docs:
            chunk = self._format_doc(doc)
            chunk_tokens = self.count_tokens(chunk)
            if used_tokens + chunk_tokens <= budget:
                assembled_chunks.append(chunk)
                selected.append(doc)
                used_tokens += chunk_tokens
                continue

            # Try partial inclusion when a full chunk does not fit.
            remaining = budget - used_tokens
            if remaining < 32:
                break
            words = doc.page_content.split()
            lo, hi = 0, len(words)
            best_text = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate_text = " ".join(words[:mid])
                candidate_chunk = f"[Source: {doc.metadata.get('source', 'unknown')}]\n{candidate_text}"
                tokens = self.count_tokens(candidate_chunk)
                if tokens <= remaining:
                    best_text = candidate_text
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best_text:
                partial_doc = Document(page_content=best_text, metadata=doc.metadata)
                partial_chunk = self._format_doc(partial_doc)
                assembled_chunks.append(partial_chunk)
                selected.append(partial_doc)
                used_tokens += self.count_tokens(partial_chunk)
            break

        context = "\n\n---\n\n".join(assembled_chunks) if assembled_chunks else "No relevant context found."
        return ContextAssemblyResult(
            context=context,
            selected_docs=selected,
            context_tokens=used_tokens,
        )
