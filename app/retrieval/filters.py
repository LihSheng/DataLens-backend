"""
Metadata filters for retrieval — filter by document ID, tags, source, etc.
"""
from typing import List, Optional

from langchain_core.documents import Document


def filter_by_document_ids(
    documents: List[Document],
    allowed_ids: List[str],
) -> List[Document]:
    """
    Keep only documents whose metadata['document_id'] is in allowed_ids.

    Args:
        documents: Input document list.
        allowed_ids: Whitelist of document IDs to retain.

    Returns:
        Filtered document list.
    """
    allowed_set = set(allowed_ids)
    return [doc for doc in documents if doc.metadata.get("document_id") in allowed_set]


def filter_by_source(
    documents: List[Document],
    sources: List[str],
) -> List[Document]:
    """
    Keep only documents from one of the specified source files.

    Args:
        documents: Input document list.
        sources: Whitelist of source filenames to retain.

    Returns:
        Filtered document list.
    """
    source_set = set(sources)
    return [doc for doc in documents if doc.metadata.get("source") in source_set]


def filter_by_tags(
    documents: List[Document],
    required_tags: List[str],
) -> List[Document]:
    """
    Keep only documents that have ALL required tags in metadata['tags'].

    Args:
        documents: Input document list.
        required_tags: Tags that must all be present in a document's metadata.

    Returns:
        Filtered document list.
    """
    def has_all_tags(doc) -> bool:
        doc_tags = set(doc.metadata.get("tags", []))
        return all(tag in doc_tags for tag in required_tags)

    return [doc for doc in documents if has_all_tags(doc)]


def apply_filters(
    documents: List[Document],
    document_ids: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> List[Document]:
    """
    Apply all specified filter functions in sequence.

    Filters are applied in order: document_ids → sources → tags.
    Any filter set to None is skipped.

    Args:
        documents: Input document list.
        document_ids: Optional whitelist of document IDs.
        sources: Optional whitelist of source filenames.
        tags: Optional list of required tags.

    Returns:
        Filtered document list after all applicable filters.
    """
    result = documents
    if document_ids:
        result = filter_by_document_ids(result, document_ids)
    if sources:
        result = filter_by_source(result, sources)
    if tags:
        result = filter_by_tags(result, tags)
    return result
