"""
Reciprocal Rank Fusion (RRF) for combining RAG and Wiki retrieval results.
"""

from typing import List, Optional

from haystack import Document

RRF_K = 60


def reciprocal_rank_fusion(
    rag_docs: List[Document],
    wiki_docs: List[Document],
    rag_weight: float = 1.0,
    wiki_weight: float = 1.0,
    top_k: Optional[int] = None,
) -> List[Document]:
    """
    Fuse RAG and Wiki document lists using Reciprocal Rank Fusion.

    Args:
        rag_docs: Documents from the RAG pipeline (dense + sparse).
        wiki_docs: Documents from the Wiki keyword/graph search.
        rag_weight: Weight applied to RAG ranks.
        wiki_weight: Weight applied to Wiki ranks.
        top_k: Maximum number of documents to return. None means no truncation.

    Returns:
        A deduplicated list of Documents sorted by RRF score descending.
    """
    scores: dict[int, float] = {}

    for rank, doc in enumerate(rag_docs, start=1):
        scores[id(doc)] = scores.get(id(doc), 0.0) + rag_weight * (
            1.0 / (RRF_K + rank)
        )

    for rank, doc in enumerate(wiki_docs, start=1):
        scores[id(doc)] = scores.get(id(doc), 0.0) + wiki_weight * (
            1.0 / (RRF_K + rank)
        )

    all_docs = list(rag_docs) + list(wiki_docs)
    all_docs.sort(key=lambda d: scores.get(id(d), 0.0), reverse=True)

    seen: set[str] = set()
    fused: List[Document] = []
    for doc in all_docs:
        key = doc.content[:200] if doc.content else ""
        if key in seen:
            continue
        seen.add(key)
        doc.score = scores.get(id(doc), 0.0)
        fused.append(doc)

    if top_k is not None and len(fused) > top_k:
        fused = fused[:top_k]

    return fused
