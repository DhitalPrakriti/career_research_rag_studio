from __future__ import annotations

import re

from rag_studio.schema import RetrievedChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


def generate_query_variants(question: str, max_queries: int = 3) -> list[str]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    original = question.strip()
    if not original:
        return []

    variants = [original]
    keywords = _keyword_query(original)
    if keywords and keywords.lower() != original.lower():
        variants.append(keywords)

    normalized = _normalized_query(original)
    if normalized.lower() not in {variant.lower() for variant in variants}:
        variants.append(normalized)

    return variants[:max_queries]


def reciprocal_rank_fusion_many(
    ranked_lists: list[list[RetrievedChunk]],
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked_results in ranked_lists:
        for rank, result in enumerate(ranked_results, start=1):
            chunk_id = result.chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk_id] = result

    sorted_ids = sorted(rrf_scores, key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
    return [
        RetrievedChunk(chunk=chunk_map[chunk_id].chunk, score=rrf_scores[chunk_id])
        for chunk_id in sorted_ids[:top_k]
    ]


def _keyword_query(question: str) -> str:
    tokens = TOKEN_RE.findall(question)
    keywords = [
        token
        for token in tokens
        if (len(token) > 2 or token.isupper()) and token.lower() not in _STOPWORDS
    ]
    return " ".join(keywords)


def _normalized_query(question: str) -> str:
    return re.sub(r"\s+", " ", question.replace("?", " ")).strip()


_STOPWORDS = {
    "about",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "the",
    "this",
    "what",
    "where",
    "which",
    "with",
    "you",
    "your",
}
