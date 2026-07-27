from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from rag_studio.schema import Chunk, RetrievedChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_counts: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._average_document_length = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a BM25 retriever with no chunks")

        self._chunks = list(chunks)
        self._term_counts = []
        self._document_frequencies = Counter()
        self._document_lengths = []

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            term_counts = Counter(tokens)
            self._term_counts.append(term_counts)
            self._document_lengths.append(len(tokens))
            self._document_frequencies.update(term_counts.keys())

        self._average_document_length = sum(self._document_lengths) / len(self._document_lengths)

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            raise RuntimeError("BM25 retriever is empty. Ingest documents before querying.")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored_results: list[RetrievedChunk] = []
        for index, chunk in enumerate(self._chunks):
            if not _metadata_matches(chunk.metadata, metadata_filter):
                continue
            score = self._score(query_terms, index)
            if score <= 0:
                continue
            scored_results.append(RetrievedChunk(chunk=chunk, score=score))

        scored_results.sort(key=lambda result: result.score, reverse=True)
        return scored_results[:top_k]

    def _score(self, query_terms: list[str], document_index: int) -> float:
        score = 0.0
        term_counts = self._term_counts[document_index]
        document_length = self._document_lengths[document_index]
        total_documents = len(self._chunks)

        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue

            document_frequency = self._document_frequencies[term]
            idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self._average_document_length
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator

        return score


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    return all(metadata.get(key) == value for key, value in metadata_filter.items())
