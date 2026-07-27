from __future__ import annotations

from typing import Any

from rag_studio.schema import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Combine dense and BM25 ranked lists using Reciprocal Rank Fusion.

    Why k=60? It dampens the influence of very top ranks so a single
    #1 result doesn't completely dominate. Standard default from the
    original RRF paper.

    Score formula: RRF(chunk) = 1/(k + rank_in_dense) + 1/(k + rank_in_bm25)
    A chunk appearing near the top of BOTH lists gets the highest score.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Score from dense results
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk.id
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk_id] = result

    # Score from BM25 results — add to existing score if chunk appeared in both
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.chunk.id
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk_id] = result

    # Sort by combined RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Rebuild RetrievedChunk list with RRF score as the new score
    return [
        RetrievedChunk(chunk=chunk_map[cid].chunk, score=rrf_scores[cid])
        for cid in sorted_ids[:top_k]
    ]