from __future__ import annotations

import hashlib
import re

from rag_studio.schema import Chunk, Document


WORD_RE = re.compile(r"\S+")


class WordChunker:
    def __init__(self, chunk_size: int = 260, chunk_overlap: int = 60) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document_index, document in enumerate(documents):
            words = WORD_RE.findall(document.text)
            if not words:
                continue

            step = self.chunk_size - self.chunk_overlap
            for chunk_index, start in enumerate(range(0, len(words), step)):
                chunk_words = words[start : start + self.chunk_size]
                if not chunk_words:
                    continue
                text = " ".join(chunk_words)
                metadata = {
                    **document.metadata,
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                    "word_start": start,
                    "word_end": start + len(chunk_words),
                }
                chunks.append(Chunk(id=_chunk_id(text, metadata), text=text, metadata=metadata))
                if start + self.chunk_size >= len(words):
                    break
        return chunks


def _chunk_id(text: str, metadata: dict) -> str:
    source = metadata.get("source_path", "")
    page = metadata.get("page", "")
    chunk_index = metadata.get("chunk_index", "")
    raw = f"{source}:{page}:{chunk_index}:{text[:120]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

