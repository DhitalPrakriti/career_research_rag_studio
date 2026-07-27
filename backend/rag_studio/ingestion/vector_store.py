from __future__ import annotations

from typing import Any

import numpy as np

from rag_studio.schema import Chunk, RetrievedChunk


class FaissVectorStore:
    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install faiss-cpu to use vector retrieval: pip install faiss-cpu") from exc

        self._faiss = faiss
        self._index = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store with no chunks")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        vectors = _as_float32_matrix(vectors)
        index = self._faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._index = index
        self._chunks = list(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("Vector store is empty. Ingest documents before querying.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_matrix = _as_float32_matrix(query_vector)
        search_k = len(self._chunks) if metadata_filter else min(top_k, len(self._chunks))
        scores, indices = self._index.search(query_matrix, search_k)
        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            if not _metadata_matches(chunk.metadata, metadata_filter):
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results


def _as_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix")
    return matrix


def _metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    return all(metadata.get(key) == value for key, value in metadata_filter.items())

