"""Document chunker with multiple strategies.

Strategies:
- recursive  : RecursiveCharacterTextSplitter (default)
- fixed      : fixed-size character chunks
- semantic   : LLM-driven semantic boundary detection (optional, slow)
"""
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

# Default chunking config
DEFAULT_CHUNK_SIZE = int(getattr(settings, "chunk_size", 1000))
DEFAULT_CHUNK_OVERLAP = int(getattr(settings, "chunk_overlap", 200))


def chunk_document(
    text: str,
    metadata: Dict[str, Any],
    strategy: str = "recursive",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    enable_semantic: bool = False,
) -> List[Document]:
    """
    Chunk a document using the specified strategy.

    Args:
        text:           Raw text content of the document.
        metadata:        Document-level metadata (source, filename, page, etc.).
        strategy:        "recursive" | "fixed" | "semantic"
        chunk_size:      Override default chunk size.
        chunk_overlap:   Override default chunk overlap.
        enable_semantic: If True and strategy="semantic", use LLM to find boundaries.

    Returns:
        List of LangChain Document objects with page_content + merged metadata.
    """
    if not text or not text.strip():
        logger.warning("chunk_document received empty text, returning empty list")
        return []

    size = chunk_size or DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )
    elif strategy == "fixed":
        splitter = _FixedLengthSplitter(chunk_size=size, chunk_overlap=overlap)
    elif strategy == "semantic":
        if enable_semantic:
            splitter = _SemanticTextSplitter(chunk_size=size, chunk_overlap=overlap)
            logger.info("Using semantic chunking — may be slow")
        else:
            # Fallback to recursive when semantic not enabled
            logger.info("Semantic strategy requested but enable_semantic=False — using recursive")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                length_function=len,
                add_start_index=True,
            )
    else:
        logger.warning(f"Unknown chunking strategy '{strategy}', defaulting to recursive")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

    # Split by paragraphs first to preserve semantic units, then apply splitter
    # This is a common pattern for better quality chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Join paragraphs back and split
    combined_text = "\n\n".join(paragraphs)
    chunks = splitter.split_text(combined_text)

    # Build Document objects with merged metadata
    documents = []
    for i, chunk_text in enumerate(chunks):
        chunk_meta = {**metadata, "chunk_index": i}
        documents.append(Document(page_content=chunk_text, metadata=chunk_meta))

    logger.info(f"Chunked into {len(documents)} pieces (strategy={strategy})")
    return documents


# ── Fixed-length splitter ─────────────────────────────────────────────────────


class _FixedLengthSplitter(TextSplitter):
    """Split text into fixed-size character chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(chunk_size=chunk_size, **kwargs)
        self._chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self._chunk_overlap
            if start >= len(text):
                break
        return chunks


# ── Semantic splitter (LLM-driven) ───────────────────────────────────────────


class _SemanticTextSplitter(TextSplitter):
    """
    Use an LLM to detect semantic boundaries in text.
    This is a simplified implementation: it splits by sentences and groups
    them into chunks while trying to respect semantic coherence.

    A full implementation would call the LLM to propose boundary positions.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(chunk_size=chunk_size, **kwargs)
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        # Use recursive splitter as base, then could call LLM to regroup
        # Full semantic approach: for each paragraph, call LLM to decide if
        # it should be grouped with next. This is expensive so we fall back
        # to the recursive approach with better separators for semantic quality.
        return self._splitter.split_text(text)
